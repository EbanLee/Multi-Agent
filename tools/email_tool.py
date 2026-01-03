import re
import html
import imaplib
import smtplib
import email
from datetime import datetime, timedelta, timezone
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

def _default_since_date(days: int = 30) -> str:
    kst = timezone(timedelta(hours=9))
    return (datetime.now(kst).date() - timedelta(days=days)).isoformat()

def _escape_gmail_quote(s: str) -> str:
    return s.replace('"', '\\"')

def _is_ascii(s: str) -> bool:
    try:
        s.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False

def _try_xgmraw(imap: imaplib.IMAP4_SSL) -> bool:
    try:
        # 결과 0이어도 OK면 "지원"으로 판단
        status, _ = imap.uid("SEARCH", "X-GM-RAW", b"in:inbox newer_than:1d")
        return status == "OK"
    except Exception:
        return False

def _try_utf8_search(imap: imaplib.IMAP4_SSL) -> bool:
    try:
        status, _ = _uid_search_utf8(imap, ["ALL", "SUBJECT"], "test")
        return status == "OK"
    except Exception:
        return False

def _detect_search_mode(imap: imaplib.IMAP4_SSL) -> str:
    """
    returns: "xgmraw" | "utf8" | "local"
    - xgmraw: Gmail X-GM-RAW 가능
    - utf8:  표준 SEARCH CHARSET UTF-8 가능(한글 서버필터 가능)
    - local: 서버 한글검색 불가 → 로컬 필터
    """
    if _try_xgmraw(imap):
        return "xgmraw"
    if _try_utf8_search(imap):
        return "utf8"
    return "local"

def _uid_search_utf8(imap: imaplib.IMAP4_SSL, criteria_tokens: list[str], text_value: str) -> tuple[str, list]:
    """
    UID SEARCH CHARSET UTF-8 <criteria...> <literal>
    - 마지막 문자열(예: 쿠팡)을 literal로 보내서 파싱 오류를 피한다.
    """
    b = text_value.encode("utf-8")
    imap.literal = b  # imaplib이 {n}\r\n<bytes> 형태로 전송
    # 마지막 토큰은 literal placeholder로 비워둬야 함
    return imap.uid("SEARCH", "CHARSET", "UTF-8", *criteria_tokens, None)


class EmailSearchTool(Tool):
    name = "search_emails"
    description="List emails (lightweight) via IMAP."
    args_schema={
        "properties": {
            "unseen_only": "bool",
            "since_date": "str (YYYY-MM-DD)",
            "before_date": "str",
            "from_contains": "str",
            "subject_contains": "str",
            "limit": "int",
            "prefetch_multiplier": "int",
        },
        "required": []
    }
    
    # args_schema={
    #     "properties": {
    #         "unseen_only": {"type": "boolean", "default": False},
    #         "since_date": {"type": "string", "description": "YYYY-MM-DD"},
    #         "before_date": {"type": "string", "description": "YYYY-MM-DD"},
    #         "from_contains": {"type": "string"},
    #         "subject_contains": {"type": "string"},
    #         "limit": {"type": "integer", "default": 1, "minimum": 1, "maximum": 10},
    #         "prefetch_multiplier": {"type": "integer", "default": 20, "minimum": 1, "maximum": 50},
    #     },
    #     "required": []
    # }

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
        limit: int = 1,
        prefetch_multiplier: int = 20,
    ):  
        """
        Gmail(X-GM-RAW) 가능하면 서버에서 한글/영어 필터링.
        아니면(비Gmail/불안정) 날짜/읽음만 서버에서 제한하고,
        from/subject는 헤더 디코딩 후 로컬 필터링으로 한방 처리.
        """

        limit = max(1, min(limit, 10))
        prefetch_multiplier = max(1, min(prefetch_multiplier, 50))
        want_max = 300

        # since_date 기본값(30일)
        if not since_date:
            since_date = _default_since_date()

        imap = imaplib.IMAP4_SSL(self.imap_host)
        try:
            imap.login(self.email_addr, self.app_password)
            imap.select("INBOX")

            mode = _detect_search_mode(imap)
            # print(f"!!!! {mode=} !!!!\n")

            if mode == "xgmraw":
                raw_parts = ["in:inbox"]
                if unseen_only:
                    raw_parts.append("is:unread")
                if since_date:
                    raw_parts.append(f"after:{since_date.replace('-', '/')}")
                if before_date:
                    raw_parts.append(f"before:{before_date.replace('-', '/')}")

                if from_contains:
                    v = _escape_gmail_quote(from_contains)
                    raw_parts.append(f'from:"{v}"')
                if subject_contains:
                    v = _escape_gmail_quote(subject_contains)
                    raw_parts.append(f'subject:"{v}"')

                raw_query = " ".join(raw_parts)

                status, data = imap.uid("SEARCH", "X-GM-RAW", raw_query.encode("utf-8"))
                if status != "OK":
                    return []
                ids = (data[0] or b"").split()

            elif mode == "utf8":
                # 표준 IMAP SEARCH CHARSET UTF-8 (문자열은 literal로 보내야 안전)
                base = ["UNSEEN" if unseen_only else "ALL"]

                if since_date:
                    base += ["SINCE", imap_date(since_date)]
                if before_date:
                    base += ["BEFORE", imap_date(before_date, before=True)]

                def run_one(key: str, value: str):
                    tokens = base + [key]
                    status, data = _uid_search_utf8(imap, tokens, value)
                    if status != "OK":
                        return set()
                    return set((data[0] or b"").split())

                ids_set = None

                if from_contains:
                    s = run_one("FROM", from_contains)
                    ids_set = s if ids_set is None else (ids_set & s)

                if subject_contains:
                    s = run_one("SUBJECT", subject_contains)
                    ids_set = s if ids_set is None else (ids_set & s)

                # from/subject 둘 다 없으면 base만으로 검색
                if ids_set is None:
                    status, data = imap.uid("SEARCH", *base)
                    if status != "OK":
                        return []
                    ids = (data[0] or b"").split()
                else:
                    ids = list(ids_set)


            else:
                # mode == "local"
                # 표준 IMAP SEARCH: 한글 필터는 넣지 말고(ASCII 오류/서버편차) 날짜/읽음만 제한
                criteria = ["UNSEEN" if unseen_only else "ALL"]

                if since_date:
                    criteria += ["SINCE", imap_date(since_date)]
                if before_date:
                    criteria += ["BEFORE", imap_date(before_date, before=True)]

                # ASCII면 서버에서 더 좁히기
                if from_contains and _is_ascii(from_contains):
                    criteria += ["FROM", f'"{from_contains}"']
                if subject_contains and _is_ascii(subject_contains):
                    criteria += ["SUBJECT", f'"{subject_contains}"']

                status, data = imap.uid("SEARCH", *criteria)
                if status != "OK":
                    return []
                ids = (data[0] or b"").split()


            if not ids:
                return []

            # UID 정렬 (오름차순)
            sorted_ids = sorted(ids, key=lambda x: int(x))

            need = limit
            if mode in ("xgmraw", "utf8"):  # 서버 필터 신뢰 → 최소 여유만
                want = min(need * 2, want_max)
            else:   # 로컬 필터 대비 → 넉넉히
                want = min(need * prefetch_multiplier, want_max)

            candidate_ids = sorted_ids[-want:] if want < len(sorted_ids) else sorted_ids

            # 헤더만 fetch 후 로컬 필터(한글/영어 한방)
            fetch_query = "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])"

            results = []
            need_count = limit

            for uid in reversed(candidate_ids):
                status_h, header_data = imap.uid("FETCH", uid, fetch_query)
                if status_h != "OK" or not header_data:
                    continue

                raw = next((it[1] for it in header_data if isinstance(it, tuple) and it[1]), None)
                if not raw:
                    continue

                msg = email.message_from_bytes(raw)
                subject = decode_mime_header(msg.get("Subject"))
                from_addr = decode_mime_header(msg.get("From"))
                date = decode_mime_header(msg.get("Date"))
                
                # print("subject: ", subject)
                # print("from_addr: ", from_addr)
                # print("date: ", date, "\n")

                # 로컬 필터: Gmail/비Gmail 공통 적용(일관성 + 안전)
                if subject_contains and subject_contains not in subject:
                    continue
                if from_contains and from_contains not in from_addr:
                    continue

                results.append({
                    "uid": uid.decode(errors="replace"),
                    "from_addr": from_addr,
                    "subject": subject,
                    "date": date,
                })

                # 충분히 모이면 조기 종료(성능)
                if len(results) >= need_count:
                    break

            return results[:limit]

        finally:
            try:
                imap.close()
            except Exception:
                pass
            try:
                imap.logout()
            except Exception:
                pass


class EmailGetTool(Tool):
    name = "get_emails"
    description = "Get emails (full) by ids via IMAP."
    args_schema = {
        "properties": {
            "ids": "list[str]",
        },
        "required": ["ids"]
    }

    # args_schema = {
    #     "properties": {
    #         "ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5},
    #         "include_body": {"type": "boolean", "default": True},
    #         "max_body_chars": {"type": "integer", "default": 3000, "minimum": 1000, "maximum": 100000},
    #     },
    #     "required": ["ids"]
    # }

    def __init__(self, email_addr: str, app_password: str, imap_host: str):
        self.email_addr = email_addr
        self.app_password = app_password
        self.imap_host = imap_host

    def __call__(self, ids: list[str], max_body_chars: int=10000):
        ids = ids[max(0, len(ids)-5):]

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

                body = extract_text_body(full_msg)
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
            "to": "list[str]",
            "cc": "list[str]",
            "subject": "str",
            "body_text": "str",
        },
        "required": ["to", "subject", "body_text"]
    }

    # args_schema={
    #     "properties": {
    #         "to": {"type": "array", "items": {"type": "string"}, "minItems": 1},
    #         "cc": {"type": "array", "items": {"type": "string"}},
    #         "subject": {"type": "string", "minLength": 1},
    #         "body_text": {"type": "string", "minLength": 1},
    #     },
    #     "required": ["to", "subject", "body_text"]
    # }

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

