import re
import html
import imaplib
import smtplib
import email
from typing import Any, Optional
from email.header import decode_header
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import parseaddr

from tools import Tool

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
            text = payload.decode(charset, errors="replace")

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


class EmailReadTool(Tool):
    name = "read_email"
    description="IMAP으로 받은편지함(INBOX)에서 메일을 검색/조회한다."
    args_schema={
        "parameters": {
            "unseen_only": {"type": "boolean", "default": False},
            "subject_contains": {"type": "string"},
            "from_contains": {"type": "string"},
            "limit": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50},
            "include_body_preview": {"type": "boolean", "default": False},
            "body_preview_chars": {"type": "integer", "default": 200, "minimum": 50, "maximum": 1000},
        },
        "required": [],
    }

    def __init__(self, email_addr: str, app_password: str, imap_host: str):
        self.email_addr = email_addr
        self.app_password = app_password
        self.imap_host = imap_host

    def __call__(
        self,
        unseen_only: bool = False,
        subject_contains: Optional[str] = None,
        from_contains: Optional[str] = None,
        limit: int = 10,
        include_body_preview: bool = False,
        body_preview_chars: int = 200,
    ):
        limit = max(1, min(limit, 50))
        
        imap = imaplib.IMAP4_SSL(self.imap_host)  # (1) SSL로 IMAP 서버 접속
        try:
            imap.login(self.email_addr, self.app_password)  # (2) 로그인
            imap.select("INBOX")                     # (3) 받은편지함 선택
            
            # (4) IMAP SEARCH 조건 구성
            criteria: list[str] = ["UNSEEN" if unseen_only else "ALL"]  # 안본거만 읽을지
            if subject_contains:
                criteria += ["SUBJECT", f"\"{subject_contains}\""]
            if from_contains:
                criteria += ["FROM", f"\"{from_contains}\""]
            
            status, data = imap.search(None, *criteria)     # 실제 검색
            ids = data[0].split()                           # 검색 결과: 메일 id 리스트(bytes)
            ids = ids[-limit:]                              # 최신 limit개
            
            # print(ids)
            
            if status != "OK":
                return []

            results = []
            for mid in ids[::-1]:           # 최근 메일부터 보기 위해 reverse
                # 헤더만 가져오기(가벼움)
                status_h, header_data = imap.fetch(mid, "(BODY.PEEK[HEADER])")
                if status_h != "OK" or not header_data:
                    continue

                # fetch 결과에서 실제 bytes payload를 찾아 파싱
                raw_header = None
                for item in header_data:
                    if isinstance(item, tuple) and item[1]:
                        raw_header = item[1]
                        break
                if not raw_header:
                    continue

                msg = email.message_from_bytes(raw_header)
                subject = decode_mime_header(msg.get("Subject"))
                from_addr = decode_mime_header(msg.get("From"))
                date = decode_mime_header(msg.get("Date"))

                # print(subject)
                # print(from_addr)
                # print(date,"\n")


                body_preview = None
                if include_body_preview:
                    # (5-2) 본문 미리보기가 필요하면 RFC822로 전체 가져와서 text/plain 추출
                    status_f, full_data = imap.fetch(mid, "(RFC822)")
                    if status_f == "OK" and full_data and isinstance(full_data[0], tuple):
                        full_msg = email.message_from_bytes(full_data[0][1])
                        body = extract_text_body(full_msg).strip()
                        body_preview = body[:body_preview_chars]

                results.append(
                    {
                        'id':mid.decode() if isinstance(mid, (bytes, bytearray)) else str(mid),
                        'from_addr':from_addr,
                        'subject':subject,
                        'date':date,
                        'body_preview':body_preview,
                    }
                )


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
        
        return results


class EmailSendTool:
    """
    [역할]
    - SMTP로 메일을 발송하는 Tool.

    [왜 SMTP?]
    - SMTP는 '메일을 보내는' 표준 프로토콜.
    """

    name="send_email",
    description="SMTP로 이메일을 발송한다. to/subject/body_text가 필수.",
    args_schema={
        "type": "object",
        "parameters": {
            "to": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "cc": {"type": "array", "items": {"type": "string"}},
            "subject": {"type": "string", "minLength": 1},
            "body_text": {"type": "string", "minLength": 1},
        },
        "required": ["to", "subject", "body_text"],
    }

    def __init__(self, email_addr: str, app_password: str, smtp_host: str, smtp_port: int):
        self.email_addr = email_addr
        self.app_password = app_password
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port

    def run(self, to: list[str], subject: str, body_text: str, cc: Optional[list[str]] = None) -> dict[str, Any]:
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

