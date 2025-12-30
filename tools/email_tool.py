import re
import html
import imaplib
import smtplib
import email
from datetime import datetime, timedelta
from typing import Any, Optional
from email.header import decode_header
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import parseaddr

from tools import Tool


def imap_date(date_yyyy_mm_dd: str, *, before: bool = False) -> str:
    dt = datetime.strptime(date_yyyy_mm_dd, "%Y-%m-%d")
    if before:
        dt += timedelta(days=1)  # BEFORE는 exclusive라 하루 보정
    return dt.strftime("%d-%b-%Y")

def decode_mime_header(value: Optional[str]) -> str:
    """
    Subject/From 같은 헤더는 '=?utf-8?B?...?=' 처럼 MIME 인코딩이 붙을 수 있음.
    이걸 사람이 읽는 문자열로 안전하게 디코딩한다.
    """
    if not value:
        return ""
    parts = decode_header(value)
    out = []
    for text, enc in parts:
        if isinstance(text, bytes):
            out.append(text.decode(enc or "utf-8", errors="replace"))
        else:
            out.append(text)
    return "".join(out).strip()


def normalize_email(addr: str) -> str:
    """
    '홍길동 <abc@gmail.com>' 같은 문자열에서 실제 이메일 주소만 추출.
    """
    _, e = parseaddr(addr)
    return e.strip().lower()

def html_to_text(html_str: str) -> str:
    # 1) HTML 엔티티(&quot; 등) 해제
    s = html.unescape(html_str)

    # 2) 줄바꿈 의미 있는 태그를 줄바꿈으로 치환
    s = re.sub(r"(?i)<\s*br\s*/?\s*>", "\n", s)
    s = re.sub(r"(?i)</\s*p\s*>", "\n", s)
    s = re.sub(r"(?i)</\s*div\s*>", "\n", s)
    s = re.sub(r"(?i)</\s*tr\s*>", "\n", s)
    s = re.sub(r"(?i)</\s*li\s*>", "\n", s)

    # 3) 나머지 태그 제거
    s = re.sub(r"<[^>]+>", " ", s)

    # 4) 공백/줄바꿈 정리
    s = re.sub(r"\r", "", s)
    s = re.sub(r"\n\s*\n+", "\n\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def extract_text_body(msg: email.message.Message) -> str:
    """
    메일 본문은 multipart(본문+첨부)인 경우가 많다.
    그중 text/plain 파트를 우선 찾아 본문 텍스트를 뽑는다.
    우선순위:
    1) text/plain (attachment 아닌 것)
    2) text/html (attachment 아닌 것) -> html_to_text로 변환
    """
    plain = None
    html_part = None

    if msg.is_multipart():
        for part in msg.walk():
            disp = part.get_content_disposition()
            if disp == "attachment":
                continue

            ctype = part.get_content_type()
            payload = part.get_payload(decode=True) or b""
            charset = part.get_content_charset() or "utf-8"
            
            try: 
                text = payload.decode(charset, errors="replace")
            except LookupError: 
                text = payload.decode("utf-8", errors="replace")

            if ctype == "text/plain" and plain is None:
                plain = text
            elif ctype == "text/html" and html_part is None:
                html_part = text
    else:
        ctype = msg.get_content_type()
        payload = msg.get_payload(decode=True) or b""
        charset = msg.get_content_charset() or "utf-8"
        text = payload.decode(charset, errors="replace")
        if ctype == "text/plain":
            plain = text
        elif ctype == "text/html":
            html_part = text

    if plain and plain.strip():
        return plain.strip()
    if html_part and html_part.strip():
        return html_to_text(html_part)

    return ""


class EmailSearchTool(Tool):
    name = "search_emails"
    description="List emails (lightweight) via IMAP."
    args_schema={
        "properties": {
            "unseen_only": {"type": "boolean", "default": False},
            "since_date": {"type": "string", "description": "YYYY-MM-DD"},
            "before_date": {"type": "string", "description": "YYYY-MM-DD"},
            "from_contains": {"type": "string"},
            "subject_contains": {"type": "string"},
            "limit": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50},
            "offset": {"type": "integer", "default": 0, "minimum": 0},
            "prefetch_multiplier": {"type": "integer", "default": 5, "minimum": 1, "maximum": 50},
        },
        "required": []
    }

    def __init__(self, email_addr: str, app_password: str, imap_host: str):
        self.email_addr = email_addr
        self.app_password = app_password
        self.imap_host = imap_host

    def __call__(
        self,
        unseen_only: bool = False,
        since_date: Optional[str] = None,
        before_date: Optional[str] = None,
        from_contains: Optional[str] = None,
        subject_contains: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
        prefetch_multiplier: int = 5,
    ):        
        limit = max(1, min(limit, 50))
        offset = max(0, offset)
        prefetch_multiplier = max(1, min(prefetch_multiplier, 50))
        
        imap = imaplib.IMAP4_SSL(self.imap_host)  # (1) SSL로 IMAP 서버 접속
        try:
            imap.login(self.email_addr, self.app_password)  # (2) 로그인
            imap.select("INBOX")                     # (3) 받은편지함 선택
            
            # (4) IMAP SEARCH 조건 구성
            criteria: list[str] = ["UNSEEN" if unseen_only else "ALL"]  # 안본거만 읽을지
            
            if since_date:
                criteria += ["SINCE", imap_date(since_date)]
            if before_date:
                criteria += ["BEFORE", imap_date(before_date, before=True)]
            
            status, data = imap.uid("search", None, *criteria)
    
            if status != "OK":
                return []
            
            ids = data[0].split()                           # 검색 결과: 메일 id 리스트(bytes)
            sorted_ids = sorted(ids, key=lambda x: int(x))
            
            # 로컬 필터 대비 넉넉히 가져오기
            want = (offset + limit) * prefetch_multiplier
            candidate_ids = sorted_ids[-want:] if want < len(sorted_ids) else sorted_ids                             # 최신 limit개
            
            # 헤더만 가져오기(가벼움)
            fetch_query = "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE MESSAGE-ID)])"

            results = []
            for uid in candidate_ids[::-1]:
                status_h, header_data = imap.uid("fetch", uid, fetch_query)
                if status_h != "OK" or not header_data:
                    continue

                raw = next((it[1] for it in header_data if isinstance(it, tuple) and it[1]), None)
                if not raw:
                    continue

                msg = email.message_from_bytes(raw)
                subject = decode_mime_header(msg.get("Subject"))
                from_addr = decode_mime_header(msg.get("From"))
                date = decode_mime_header(msg.get("Date"))
                message_id = decode_mime_header(msg.get("Message-ID"))

                print("Subject: ", subject)
                print("From_addr: ",from_addr)
                print("Date: ",date,"\n")

                # 로컬 필터 (한글처리)
                if subject_contains and subject_contains not in subject:
                    continue
                if from_contains and from_contains not in from_addr:
                    continue

                results.append({
                    "uid": uid.decode(),
                    "message_id": message_id,
                    "from_addr": from_addr,
                    "subject": subject,
                    "date": date,
                })

            return results[offset: offset + limit]

        finally:
            # (6) 세션 종료(안 하면 연결이 쌓일 수 있음)
            try:
                imap.close()
            except Exception:
                pass
            try:
                imap.logout()
            except Exception:
                pass
        

class EmailGetTool(Tool):
    name = "get_email"
    description = "Get a single email (full) by id via IMAP."
    args_schema = {
        "properties": {
            "uid": {"type": "string", "minLength": 1},
            "include_body": {"type": "boolean", "default": True},
            "max_body_chars": {"type": "integer", "default": 2000, "minimum": 100, "maximum": 100000},
        },
        "required": ["id"]
    }

    def __init__(self, email_addr: str, app_password: str, imap_host: str):
        self.email_addr = email_addr
        self.app_password = app_password
        self.imap_host = imap_host

    def __call__(self, uid: str, include_body: bool = True, max_body_chars: int = 2000):
        imap = imaplib.IMAP4_SSL(self.imap_host)
        try:
            imap.login(self.email_addr, self.app_password)
            imap.select("INBOX")

            uid = uid.encode()
            typ, full_data = imap.uid("fetch", uid, "(RFC822)")
            if typ != "OK" or not full_data or not isinstance(full_data[0], tuple):
                return {"error": "fetch_failed", "id": id}

            full_msg = email.message_from_bytes(full_data[0][1])
            subject = decode_mime_header(full_msg.get("Subject"))
            from_addr = decode_mime_header(full_msg.get("From"))
            to_addr = decode_mime_header(full_msg.get("To"))
            date = decode_mime_header(full_msg.get("Date"))

            body = extract_text_body(full_msg) if include_body else ""
            if len(body) > max_body_chars:
                body = body[:max_body_chars]

            return {
                "id": id,
                "from_addr": from_addr,
                "to": to_addr,
                "subject": subject,
                "date": date,
                "body": body,
            }

        finally:
            try: imap.close()
            except Exception: pass
            try: imap.logout()
            except Exception: pass


class EmailGetBatchTool(Tool):
    name = "get_emails"
    description = "Get multiple emails (full) by ids via IMAP (UID)."
    args_schema = {
        "properties": {
            "ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 50},
            "include_body": {"type": "boolean", "default": True},
            "max_body_chars": {"type": "integer", "default": 2000, "minimum": 100, "maximum": 100000},
        },
        "required": ["ids"]
    }

    def __init__(self, email_addr: str, app_password: str, imap_host: str):
        self.email_addr = email_addr
        self.app_password = app_password
        self.imap_host = imap_host

    def __call__(self, ids: list[str], include_body: bool=True, max_body_chars: int=2000):
        imap = imaplib.IMAP4_SSL(self.imap_host)
        try:
            imap.login(self.email_addr, self.app_password)
            imap.select("INBOX")

            out = []
            for id in ids:
                uid = id.encode()
                typ, full_data = imap.uid("fetch", uid, "(RFC822)")
                if typ != "OK" or not full_data or not isinstance(full_data[0], tuple):
                    out.append({"error": "fetch_failed", "id": id})
                    continue

                full_msg = email.message_from_bytes(full_data[0][1])
                subject = decode_mime_header(full_msg.get("Subject"))
                from_addr = decode_mime_header(full_msg.get("From"))
                to_addr = decode_mime_header(full_msg.get("To"))
                date = decode_mime_header(full_msg.get("Date"))

                body = extract_text_body(full_msg) if include_body else ""
                if len(body) > max_body_chars:
                    body = body[:max_body_chars]

                out.append({
                    "id": id,
                    "from_addr": from_addr,
                    "to": to_addr,
                    "subject": subject,
                    "date": date,
                    "body": body,
                })

            return out

        finally:
            try: imap.close()
            except Exception: pass
            try: imap.logout()
            except Exception: pass



class EmailSendTool:
    """
    [역할]
    - SMTP로 메일을 발송하는 Tool.

    [왜 SMTP?]
    - SMTP는 '메일을 보내는' 표준 프로토콜.
    """

    name="send_email"
    description="Send email via SMTP."
    args_schema={
        "properties": {
            "to": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "cc": {"type": "array", "items": {"type": "string"}},
            "subject": {"type": "string", "minLength": 1},
            "body_text": {"type": "string", "minLength": 1},
        },
        "required": ["to", "subject", "body_text"]
    }

    def __init__(self, email_addr: str, app_password: str, smtp_host: str, smtp_port: int):
        self.email_addr = email_addr
        self.app_password = app_password
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port

    def __call__(self, to: list[str], subject: str, body_text: str, cc: Optional[list[str]] = None) -> dict[str, Any]:
        """
        [흐름]
        1) MIME 메시지(봉투) 생성: From/To/Subject 설정
        2) 본문을 MIMEText로 붙임
        3) SMTP 접속 → TLS → 로그인 → sendmail

        NOTE:
        - 첨부가 필요하면 MIMEMultipart에 attachment 파트를 추가하면 됨.
        """
        cc = cc or []

        # (1) 멀티파트 메일 컨테이너(본문/첨부를 담는 봉투)
        msg = MIMEMultipart()
        msg["From"] = self.email_addr
        msg["To"] = ", ".join(to)
        if cc:
            msg["Cc"] = ", ".join(cc)
        msg["Subject"] = subject

        # (2) 텍스트 본문 파트 생성 후 봉투에 attach
        msg.attach(MIMEText(body_text, "plain", "utf-8"))

        # (3) 실제 전송 대상 목록(to + cc)
        rcpt = list(to) + list(cc)

        # (4) SMTP 연결 및 발송
        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            server.starttls()  # 평문 → TLS(암호화) 전환
            server.login(self.email_addr, self.app_password)
            server.sendmail(self.email_addr, rcpt, msg.as_string())

        return {"sent_to": rcpt, "subject": subject}

