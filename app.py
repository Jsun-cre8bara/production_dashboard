import base64
import datetime as dt
import hashlib
import json
import os
import random
import secrets
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    create_engine,
    func,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, joinedload


PURCHASE_TYPES = ["지정", "비지정", "초대", "기타"]
GENDERS = ["남성", "여성", "기타"]

def _normalize_database_url(raw_url: Optional[str]) -> str:
    if not raw_url:
        return "sqlite:///./production_dashboard.db"
    if raw_url.startswith("postgres://"):
        return raw_url.replace("postgres://", "postgresql+psycopg://", 1)
    return raw_url


DATABASE_URL = _normalize_database_url(os.getenv("DATABASE_URL"))
engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,  # keep objects usable after session closes
    future=True,
)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    company_name = Column(String(255))
    role = Column(String(50), nullable=False)
    password_hash = Column(String(128), nullable=False)
    password_salt = Column(String(32), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())

    works = relationship("Work", back_populates="producer")
    campaigns = relationship("DMCampaign", back_populates="producer")


class Work(Base):
    __tablename__ = "works"

    id = Column(Integer, primary_key=True)
    producer_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String(255), nullable=False)
    genre = Column(String(120))
    description = Column(Text)
    poster_image = Column(LargeBinary)
    poster_mime = Column(String(50))
    schedule_start = Column(Date, nullable=False)
    schedule_end = Column(Date, nullable=False)
    status = Column(String(50), default="pending_review")
    reviewer_note = Column(Text)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())

    producer = relationship("User", back_populates="works")
    revisions = relationship("WorkRevision", back_populates="work", cascade="all, delete-orphan")
    campaigns = relationship("DMCampaign", back_populates="work")


class WorkRevision(Base):
    __tablename__ = "work_revisions"

    id = Column(Integer, primary_key=True)
    work_id = Column(Integer, ForeignKey("works.id"), nullable=False)
    payload = Column(Text, nullable=False)
    status = Column(String(50), default="pending")
    submitter_note = Column(Text)
    reviewer_note = Column(Text)
    submitted_at = Column(DateTime, server_default=func.now())
    reviewed_at = Column(DateTime)
    reviewer_id = Column(Integer, ForeignKey("users.id"))

    work = relationship("Work", back_populates="revisions")
    reviewer = relationship("User")


class AudienceMember(Base):
    __tablename__ = "audience_members"

    id = Column(Integer, primary_key=True)
    full_name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    age = Column(Integer)
    gender = Column(String(50))
    marketing_opt_in = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())

    viewings = relationship("AudienceViewing", back_populates="audience", cascade="all, delete-orphan")


class AudienceViewing(Base):
    __tablename__ = "audience_viewings"

    id = Column(Integer, primary_key=True)
    audience_id = Column(Integer, ForeignKey("audience_members.id"), nullable=False)
    work_id = Column(Integer, ForeignKey("works.id"))
    viewing_date = Column(Date, nullable=False)
    purchase_type = Column(String(50), nullable=False)

    audience = relationship("AudienceMember", back_populates="viewings")


class DMCampaign(Base):
    __tablename__ = "dm_campaigns"

    id = Column(Integer, primary_key=True)
    work_id = Column(Integer, ForeignKey("works.id"), nullable=False)
    producer_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    subject = Column(String(255), nullable=False)
    body = Column(Text, nullable=False)
    cta_url = Column(String(500))
    scheduled_for = Column(DateTime)
    status = Column(String(50), default="draft")
    target_filters = Column(Text)
    target_audience_ids = Column(Text)
    expected_recipients = Column(Integer, default=0)
    image_bytes = Column(LargeBinary)
    image_mime = Column(String(50))
    work_snapshot = Column(Text)
    reviewer_note = Column(Text)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())

    producer = relationship("User", back_populates="campaigns")
    work = relationship("Work", back_populates="campaigns")
    logs = relationship("DMSendLog", back_populates="campaign", cascade="all, delete-orphan")


class DMSendLog(Base):
    __tablename__ = "dm_send_logs"

    id = Column(Integer, primary_key=True)
    campaign_id = Column(Integer, ForeignKey("dm_campaigns.id"), nullable=False)
    audience_id = Column(Integer, ForeignKey("audience_members.id"), nullable=False)
    status = Column(String(50), default="pending")
    sent_at = Column(DateTime)

    campaign = relationship("DMCampaign", back_populates="logs")

def _hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    salt = salt or secrets.token_hex(8)
    digest = hashlib.sha256(f"{salt}{password}".encode("utf-8")).hexdigest()
    return digest, salt


def _create_user(
    session,
    *,
    email: str,
    password: str,
    role: str,
    name: str,
    company_name: Optional[str] = None,
) -> User:
    hashed, salt = _hash_password(password)
    user = User(
        email=email.lower(),
        password_hash=hashed,
        password_salt=salt,
        role=role,
        name=name,
        company_name=company_name,
    )
    session.add(user)
    session.flush()
    return user


def _ensure_seed_data(session) -> None:
    admin = session.query(User).filter_by(email="admin@example.com").first()
    if not admin:
        admin = _create_user(
            session,
            email="admin@example.com",
            password="admin123",
            role="admin",
            name="관리자",
        )

    producer = session.query(User).filter_by(email="producer@example.com").first()
    if not producer:
        producer = _create_user(
            session,
            email="producer@example.com",
            password="producer123",
            role="producer",
            name="샘플 제작사",
            company_name="샘플 컴퍼니",
        )

    if producer and not session.query(Work).filter(Work.producer_id == producer.id).first():
        work = Work(
            producer_id=producer.id,
            title="샘플 뮤지컬",
            genre="뮤지컬",
            description="샘플 설명입니다.",
            schedule_start=dt.date.today(),
            schedule_end=dt.date.today() + dt.timedelta(days=30),
            status="live",
        )
        session.add(work)

    if not session.query(AudienceMember).first():
        sample_names = [
            "김지훈",
            "이서연",
            "박민준",
            "최수빈",
            "정다은",
            "한유진",
            "윤하린",
            "오예준",
            "배시우",
            "문채원",
        ]
        members: List[AudienceMember] = []
        for idx, name in enumerate(sample_names, start=1):
            member = AudienceMember(
                full_name=name,
                email=f"user{idx}@example.com",
                age=random.randint(18, 55),
                gender=random.choice(GENDERS[:-1]),
            )
            session.add(member)
            members.append(member)
        session.flush()

        works = session.query(Work).all()
        for member in members:
            for _ in range(random.randint(1, 5)):
                viewing = AudienceViewing(
                    audience_id=member.id,
                    work_id=random.choice(works).id if works else None,
                    purchase_type=random.choice(PURCHASE_TYPES),
                    viewing_date=dt.date.today() - dt.timedelta(days=random.randint(1, 120)),
                )
                session.add(viewing)

def init_db() -> None:
    Base.metadata.create_all(engine)
    with SessionLocal() as session:
        _ensure_seed_data(session)
        session.commit()


@contextmanager
def session_scope():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def authenticate(email: str, password: str) -> Optional[User]:
    with session_scope() as session:
        user = session.query(User).filter(User.email == email.lower()).first()
        if not user:
            return None
        hashed, _ = _hash_password(password, user.password_salt)
        if hashed == user.password_hash:
            session.expunge(user)
            return user
        return None


def image_to_data_uri(image_bytes: Optional[bytes], mime: Optional[str]) -> Optional[str]:
    if not image_bytes or not mime:
        return None
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def rerun_app():
    """Streamlit rerun helper for legacy/newer versions."""
    if hasattr(st, "rerun"):
        st.rerun()
    else:  # pragma: no cover - older Streamlit fallback
        st.experimental_rerun()


def _format_display_value(value: Optional[str]) -> str:
    if value is None:
        return "-"
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    return str(value)


def refresh_work_statuses(session, producer_id: Optional[int] = None) -> None:
    today = dt.date.today()
    query = session.query(Work).filter(Work.schedule_end < today, Work.status == "live")
    if producer_id:
        query = query.filter(Work.producer_id == producer_id)
    for work in query.all():
        work.status = "history"

def query_audience(
    session,
    *,
    age_range: Tuple[int, int],
    genders: Sequence[str],
    view_range: Tuple[int, int],
    purchase_types: Sequence[str],
    start_date: Optional[dt.date],
    end_date: Optional[dt.date],
) -> pd.DataFrame:
    start_date = start_date or (dt.date.today() - dt.timedelta(days=90))
    end_date = end_date or dt.date.today()
    view_filters = (
        session.query(
            AudienceViewing.audience_id.label("audience_id"),
            func.count(AudienceViewing.id).label("view_count"),
        )
        .filter(AudienceViewing.viewing_date.between(start_date, end_date))
    )
    if purchase_types:
        view_filters = view_filters.filter(AudienceViewing.purchase_type.in_(purchase_types))
    view_filters = view_filters.group_by(AudienceViewing.audience_id).subquery()

    query = (
        session.query(
            AudienceMember,
            func.coalesce(view_filters.c.view_count, 0).label("view_count"),
        )
        .outerjoin(view_filters, AudienceMember.id == view_filters.c.audience_id)
        .filter(
            AudienceMember.age >= age_range[0],
            AudienceMember.age <= age_range[1],
        )
    )

    if genders:
        query = query.filter(AudienceMember.gender.in_(genders))

    min_view, max_view = view_range
    query = query.filter(func.coalesce(view_filters.c.view_count, 0) >= min_view)
    query = query.filter(func.coalesce(view_filters.c.view_count, 0) <= max_view)

    rows = []
    for member, view_count in query.all():
        rows.append(
            {
                "audience_id": member.id,
                "이름": member.full_name,
                "이메일": member.email,
                "나이": member.age,
                "성별": member.gender,
                "기간 관람수": int(view_count),
                "마케팅 동의": "동의" if member.marketing_opt_in else "거부",
            }
        )
    return pd.DataFrame(rows)


def summarize_purchase_types(
    session,
    audience_ids: Sequence[int],
    purchase_types: Sequence[str],
    start_date: Optional[dt.date],
    end_date: Optional[dt.date],
) -> List[Tuple[str, int]]:
    if not audience_ids:
        return []
    query = session.query(
        AudienceViewing.purchase_type,
        func.count(AudienceViewing.id),
    ).filter(AudienceViewing.audience_id.in_(audience_ids))
    if start_date and end_date:
        query = query.filter(AudienceViewing.viewing_date.between(start_date, end_date))
    if purchase_types:
        query = query.filter(AudienceViewing.purchase_type.in_(purchase_types))
    query = query.group_by(AudienceViewing.purchase_type)
    return query.all()

def render_login():
    st.title("제작사 대시보드")
    st.caption("Streamlit + SQLAlchemy 기반 DM 타겟팅 시스템")
    with st.form("login_form"):
        email = st.text_input("이메일")
        password = st.text_input("비밀번호", type="password")
        submitted = st.form_submit_button("로그인")
        if submitted:
            user = authenticate(email, password)
            if user:
                st.session_state["user"] = {"id": user.id, "role": user.role, "name": user.name}
                st.success("로그인 성공")
                rerun_app()
            else:
                st.error("로그인 정보를 확인해주세요.")
    st.info("기본 계정 - 관리자: admin@example.com / admin123, 제작사: producer@example.com / producer123")


def get_current_user(session) -> Optional[User]:
    user_meta = st.session_state.get("user")
    if not user_meta:
        return None
    return session.query(User).get(user_meta["id"])


def render_producer_portal(user: User):
    st.sidebar.success(f"{user.company_name or user.name} 님")
    if st.sidebar.button("로그아웃"):
        st.session_state.pop("user", None)
        rerun_app()
    section = st.sidebar.radio("메뉴", ["작품 관리", "관객 타겟팅", "DM 캠페인"])
    with session_scope() as session:
        refresh_work_statuses(session, producer_id=user.id)
    if section == "작품 관리":
        render_work_management(user)
    elif section == "관객 타겟팅":
        render_audience_lab(user)
    else:
        render_campaigns(user)


def render_admin_portal(user: User):
    st.sidebar.info(f"관리자: {user.name}")
    if st.sidebar.button("로그아웃"):
        st.session_state.pop("user", None)
        rerun_app()
    section = st.sidebar.radio("관리 콘솔", ["승인 센터", "DM 모니터링"])
    if section == "승인 센터":
        render_admin_approvals()
    else:
        render_admin_dm_monitor()

def render_work_management(user: User):
    st.header("작품 관리")
    with session_scope() as session:
        works = session.query(Work).filter(Work.producer_id == user.id).order_by(Work.created_at.desc()).all()

    with st.expander("새 작품 등록 요청", expanded=False):
        with st.form("new_work"):
            title = st.text_input("작품명")
            genre = st.text_input("장르")
            description = st.text_area("작품 설명")
            poster_file = st.file_uploader("포스터 이미지 (선택)", type=["png", "jpg", "jpeg"])
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("공연 시작일", value=dt.date.today())
            with col2:
                end_date = st.date_input("공연 종료일", value=dt.date.today() + dt.timedelta(days=30))
            submit = st.form_submit_button("등록 요청")
            if submit:
                if not title:
                    st.error("작품명을 입력해주세요.")
                elif start_date > end_date:
                    st.error("공연 일정이 올바르지 않습니다.")
                else:
                    with session_scope() as session:
                        work = Work(
                            producer_id=user.id,
                            title=title,
                            genre=genre,
                            description=description,
                            schedule_start=start_date,
                            schedule_end=end_date,
                            status="pending_review",
                        )
                        if poster_file:
                            work.poster_image = poster_file.getvalue()
                            work.poster_mime = poster_file.type
                        session.add(work)
                    st.success("등록 요청이 전송되었습니다. 관리자 승인 후 노출됩니다.")
                    rerun_app()

    live_tab, history_tab = st.tabs(["운영중 / 대기", "히스토리"])
    with live_tab:
        for work in [w for w in works if w.status in ("pending_review", "live")]:
            poster_data = image_to_data_uri(work.poster_image, work.poster_mime)
            with st.container():
                cols = st.columns([2, 1])
                with cols[0]:
                    st.subheader(f"{work.title} ({work.status})")
                    st.markdown(f"- 장르: {work.genre or '-'}")
                    st.markdown(f"- 기간: {work.schedule_start} ~ {work.schedule_end}")
                    st.markdown(work.description or "설명이 없습니다.")
                    if work.reviewer_note:
                        st.warning(f"관리자 코멘트: {work.reviewer_note}")
                with cols[1]:
                    if poster_data:
                        st.image(poster_data, use_column_width=True)
                if work.status == "live":
                    with st.expander("수정 요청"):
                        with st.form(f"rev_{work.id}"):
                            new_title = st.text_input("제목", value=work.title)
                            new_genre = st.text_input("장르", value=work.genre)
                            new_description = st.text_area("설명", value=work.description)
                            new_poster = st.file_uploader("새 포스터 (선택)", type=["png", "jpg", "jpeg"], key=f"poster_{work.id}")
                            col1, col2 = st.columns(2)
                            with col1:
                                new_start = st.date_input("시작일", value=work.schedule_start, key=f"start_{work.id}")
                            with col2:
                                new_end = st.date_input("종료일", value=work.schedule_end, key=f"end_{work.id}")
                            note = st.text_area("변경 사유", key=f"note_{work.id}")
                            submit = st.form_submit_button("수정 승인 요청")
                            if submit:
                                payload = {
                                    "title": new_title,
                                    "genre": new_genre,
                                    "description": new_description,
                                    "schedule_start": new_start.isoformat(),
                                    "schedule_end": new_end.isoformat(),
                                }
                                if new_poster:
                                    payload["poster_image"] = base64.b64encode(new_poster.getvalue()).decode("utf-8")
                                    payload["poster_mime"] = new_poster.type
                                with session_scope() as session:
                                    revision = WorkRevision(
                                        work_id=work.id,
                                        payload=json.dumps(payload, ensure_ascii=False),
                                        submitter_note=note,
                                        status="pending",
                                    )
                                    session.add(revision)
                                st.success("수정 요청이 접수되었습니다. 관리자 검수 후 적용됩니다.")
                                rerun_app()
    with history_tab:
        history = [w for w in works if w.status == "history"]
        if not history:
            st.info("히스토리에 저장된 작품이 없습니다.")
        for work in history:
            st.markdown(f"**{work.title}** ({work.schedule_start} ~ {work.schedule_end})")

def render_audience_lab(user: User):
    st.header("관객 세그먼트 & DM")
    default_start = dt.date.today() - dt.timedelta(days=90)
    default_end = dt.date.today()
    with st.form("audience_filters"):
        age_range = st.slider("연령대", value=(18, 55), min_value=10, max_value=80)
        genders = st.multiselect("성별", options=GENDERS[:-1], default=GENDERS[:-1])
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("관람 시작일", value=default_start)
        with col2:
            end_date = st.date_input("관람 종료일", value=default_end)
        view_range = st.slider("기간 내 관람 횟수", value=(0, 5), min_value=0, max_value=20)
        purchase_types = st.multiselect("구매 형태", options=PURCHASE_TYPES, default=PURCHASE_TYPES)
        submitted = st.form_submit_button("조건 적용")
        if submitted:
            st.session_state["active_filters"] = {
                "age_range": age_range,
                "genders": genders,
                "view_range": view_range,
                "start_date": start_date,
                "end_date": end_date,
                "purchase_types": purchase_types,
            }

    filters = st.session_state.get("active_filters")
    if not filters:
        st.info("왼쪽 조건을 설정하면 결과가 표시됩니다.")
        return

    with session_scope() as session:
        df = query_audience(session, **filters)
        audience_ids = df["audience_id"].tolist() if not df.empty else []
        purchase_summary = summarize_purchase_types(
            session,
            audience_ids,
            filters["purchase_types"],
            filters["start_date"],
            filters["end_date"],
        )

    if df.empty:
        st.warning("조건에 맞는 관객이 없습니다. 필터를 조정해주세요.")
        return

    st.subheader(f"필터 결과 ({len(df)}명)")
    st.dataframe(df.drop(columns=["audience_id"]))

    fig, ax = plt.subplots()
    labels = [label for label, _ in purchase_summary] or PURCHASE_TYPES
    sizes = [count for _, count in purchase_summary] or [1] * len(labels)
    ax.pie(sizes, labels=labels, autopct="%1.1f%%")
    ax.set_title("구매 형태 분포")
    st.pyplot(fig)

    if st.button("DM 보내기", key="dm_start"):
        st.session_state["dm_form_open"] = True
        st.session_state["dm_targets"] = {
            "audience_ids": audience_ids,
            "filters": filters,
            "count": len(df),
        }

    if st.session_state.get("dm_form_open"):
        render_dm_form(user)

def render_dm_form(user: User):
    target_meta = st.session_state.get("dm_targets")
    if not target_meta:
        st.info("필터를 먼저 적용해주세요.")
        return

    st.divider()
    st.subheader("DM 작성")
    st.caption(f"선택된 타겟: {target_meta['count']}명")
    with session_scope() as session:
        works = session.query(Work).filter(Work.producer_id == user.id, Work.status == "live").all()
    if not works:
        st.warning("운영중인 작품이 없습니다.")
        return

    work_options = {f"{w.title} ({w.schedule_start}~{w.schedule_end})": w.id for w in works}
    with st.form("dm_form"):
        selected_label = st.selectbox("작품 선택", options=list(work_options.keys()))
        subject = st.text_input("DM 제목")
        body = st.text_area("DM 본문", height=200, help="{{관객이름}} 같은 템플릿 변수를 활용하세요.")
        cta_url = st.text_input("CTA 링크 (선택)")
        use_default_image = st.checkbox("작품 포스터 사용", value=True)
        uploaded_image = None
        if not use_default_image:
            uploaded_image = st.file_uploader("DM 이미지 업로드", type=["png", "jpg", "jpeg"])
        schedule_mode = st.radio("발송 일정", options=["즉시", "예약"])
        scheduled_for = None
        if schedule_mode == "예약":
            scheduled_date = st.date_input("예약 날짜", value=dt.date.today() + dt.timedelta(days=1))
            schedule_time = st.time_input("예약 시간", value=dt.time(hour=10, minute=0))
            scheduled_for = dt.datetime.combine(scheduled_date, schedule_time)
        else:
            scheduled_for = dt.datetime.utcnow()
        submit = st.form_submit_button("관리자 승인 요청")
        if submit:
            work_id = work_options[selected_label]
            with session_scope() as session:
                work = session.query(Work).get(work_id)
                if not work:
                    st.error("작품을 찾을 수 없습니다.")
                    return
                image_bytes = None
                image_mime = None
                if use_default_image and work.poster_image:
                    image_bytes = work.poster_image
                    image_mime = work.poster_mime
                elif uploaded_image:
                    image_bytes = uploaded_image.getvalue()
                    image_mime = uploaded_image.type

                campaign = DMCampaign(
                    work_id=work_id,
                    producer_id=user.id,
                    subject=subject,
                    body=body,
                    cta_url=cta_url,
                    scheduled_for=scheduled_for,
                    status="pending_review",
                    target_filters=json.dumps(target_meta["filters"], default=str, ensure_ascii=False),
                    target_audience_ids=json.dumps(target_meta["audience_ids"]),
                    expected_recipients=target_meta["count"],
                    image_bytes=image_bytes,
                    image_mime=image_mime,
                    work_snapshot=json.dumps(
                        {
                            "title": work.title,
                            "schedule": f"{work.schedule_start}~{work.schedule_end}",
                            "genre": work.genre,
                        },
                        ensure_ascii=False,
                    ),
                )
                session.add(campaign)
            st.success("DM 승인 요청이 전송되었습니다.")
            st.session_state["dm_form_open"] = False
            st.session_state["dm_targets"] = None

def render_campaigns(user: User):
    st.header("DM 캠페인 현황")
    with session_scope() as session:
        campaigns = (
            session.query(DMCampaign)
            .filter(DMCampaign.producer_id == user.id)
            .options(joinedload(DMCampaign.work), joinedload(DMCampaign.logs))
            .order_by(DMCampaign.created_at.desc())
            .all()
        )
    if not campaigns:
        st.info("등록된 DM 캠페인이 없습니다.")
        return
    data = []
    for campaign in campaigns:
        data.append(
            {
                "ID": campaign.id,
                "작품": campaign.work.title if campaign.work else "-",
                "제목": campaign.subject,
                "상태": campaign.status,
                "발송 예정": campaign.scheduled_for,
                "타겟 수": campaign.expected_recipients,
            }
        )
    st.dataframe(pd.DataFrame(data))

    selected_id = st.selectbox("상세보기", options=[c.id for c in campaigns])
    campaign = next(c for c in campaigns if c.id == selected_id)
    st.subheader(f"{campaign.subject} (상태: {campaign.status})")
    st.write(campaign.body)
    if campaign.image_bytes:
        st.image(image_to_data_uri(campaign.image_bytes, campaign.image_mime))
    st.markdown(f"- 작업자 노트: {campaign.reviewer_note or '없음'}")
    st.markdown(f"- 예매 전환 로그 수: {len(campaign.logs)}")

def render_admin_approvals():
    st.header("승인 센터")
    with session_scope() as session:
        pending_works = (
            session.query(Work)
            .filter(Work.status == "pending_review")
            .options(joinedload(Work.producer))
            .all()
        )
        pending_revisions = (
            session.query(WorkRevision)
            .filter(WorkRevision.status == "pending")
            .options(joinedload(WorkRevision.work))
            .all()
        )

    st.subheader("신규 작품 승인")
    if not pending_works:
        st.caption("대기중인 신규 작품이 없습니다.")
    for work in pending_works:
        with st.expander(f"{work.title} - {work.producer.company_name or work.producer.name}", expanded=False):
            st.markdown(work.description or "-")
            st.markdown(f"- 기간: {work.schedule_start} ~ {work.schedule_end}")
            decision = st.selectbox("결정", options=["승인", "반려"], key=f"work_decision_{work.id}")
            note = st.text_area("코멘트", key=f"work_note_{work.id}")
            if st.button("처리", key=f"work_btn_{work.id}"):
                with session_scope() as session:
                    current = session.query(Work).get(work.id)
                    if current:
                        current.status = "live" if decision == "승인" else "rejected"
                        current.reviewer_note = note
                st.success("처리 완료")
                rerun_app()

    st.subheader("수정 요청")
    if not pending_revisions:
        st.caption("대기중인 수정 요청이 없습니다.")
    for revision in pending_revisions:
        payload = json.loads(revision.payload)
        with st.expander(f"{revision.work.title} 수정 요청 #{revision.id}", expanded=False):
            field_labels = {
                "title": "제목",
                "genre": "장르",
                "description": "설명",
                "schedule_start": "시작일",
                "schedule_end": "종료일",
            }
            changes = []
            for field, label in field_labels.items():
                requested = payload.get(field)
                current = getattr(revision.work, field, None)
                if requested is None and current is None:
                    continue
                display_requested = requested
                if field in ("schedule_start", "schedule_end") and requested:
                    try:
                        display_requested = dt.date.fromisoformat(requested).isoformat()
                    except ValueError:
                        pass
                changes.append(
                    {
                        "필드": label,
                        "기존 값": _format_display_value(current),
                        "요청 값": display_requested or "-",
                    }
                )
            if changes:
                st.table(pd.DataFrame(changes))
            else:
                st.info("텍스트 변경 사항이 없습니다.")
            encoded_poster = payload.get("poster_image")
            if encoded_poster:
                st.image(image_to_data_uri(base64.b64decode(encoded_poster), payload.get("poster_mime")))
            if revision.submitter_note:
                st.caption(f"제작사 메모: {revision.submitter_note}")
            decision = st.selectbox("결정", options=["승인", "반려"], key=f"rev_decision_{revision.id}")
            note = st.text_area("코멘트", key=f"rev_note_{revision.id}")
            if st.button("처리", key=f"rev_btn_{revision.id}"):
                with session_scope() as session:
                    rev = session.query(WorkRevision).get(revision.id)
                    if rev:
                        rev.status = "approved" if decision == "승인" else "rejected"
                        rev.reviewed_at = dt.datetime.utcnow()
                        rev.reviewer_note = note
                        if decision == "승인":
                            work = session.query(Work).get(rev.work_id)
                            if work:
                                work.title = payload.get("title", work.title)
                                work.genre = payload.get("genre", work.genre)
                                work.description = payload.get("description", work.description)
                                if payload.get("schedule_start"):
                                    work.schedule_start = dt.date.fromisoformat(payload["schedule_start"])
                                if payload.get("schedule_end"):
                                    work.schedule_end = dt.date.fromisoformat(payload["schedule_end"])
                                if payload.get("poster_image"):
                                    work.poster_image = base64.b64decode(payload["poster_image"])
                                    work.poster_mime = payload.get("poster_mime")
                st.success("수정 요청을 처리했습니다.")
                rerun_app()

    st.subheader("DM 승인 대기")
    with session_scope() as session:
        pending_dm = (
            session.query(DMCampaign)
            .filter(DMCampaign.status == "pending_review")
            .options(joinedload(DMCampaign.work))
            .all()
        )

    if not pending_dm:
        st.caption("대기중인 DM 캠페인이 없습니다.")
    for campaign in pending_dm:
        with st.expander(f"DM #{campaign.id} - {campaign.subject}", expanded=False):
            st.markdown(f"- 작품: {campaign.work.title if campaign.work else '-'}")
            st.write(campaign.body)
            filters = json.loads(campaign.target_filters or "{}")
            filter_labels = {
                "age_range": "연령대",
                "genders": "성별",
                "view_range": "기간 관람수",
                "start_date": "기간 시작",
                "end_date": "기간 종료",
                "purchase_types": "구매 형태",
            }
            rows = []
            for key, label in filter_labels.items():
                value = filters.get(key)
                if value is None:
                    continue
                display = value
                if isinstance(value, list):
                    display = ", ".join(str(v) for v in value)
                rows.append({"조건": label, "설정값": display})
            if rows:
                st.table(pd.DataFrame(rows))
            decision = st.selectbox("결정", ["승인", "반려"], key=f"dm_decision_{campaign.id}")
            note = st.text_area("코멘트", key=f"dm_note_{campaign.id}")
            if st.button("DM 처리", key=f"dm_btn_{campaign.id}"):
                with session_scope() as session:
                    record = session.query(DMCampaign).get(campaign.id)
                    if record:
                        if decision == "승인":
                            record.status = "approved"
                        else:
                            record.status = "rejected"
                        record.reviewer_note = note
                st.success("DM 요청을 처리했습니다.")
                rerun_app()

def render_admin_dm_monitor():
    st.header("DM 모니터링")
    with session_scope() as session:
        campaigns = session.query(DMCampaign).order_by(DMCampaign.created_at.desc()).all()
    if not campaigns:
        st.info("DM 캠페인이 없습니다.")
        return
    data = []
    for campaign in campaigns:
        data.append(
            {
                "ID": campaign.id,
                "제작사": campaign.producer.company_name if campaign.producer else "-",
                "작품": campaign.work.title if campaign.work else "-",
                "상태": campaign.status,
                "타겟 수": campaign.expected_recipients,
                "예약": campaign.scheduled_for,
            }
        )
    st.dataframe(pd.DataFrame(data))

def main():
    st.set_page_config(page_title="제작사 대시보드", layout="wide")
    init_db()
    if "user" not in st.session_state:
        render_login()
        return
    with session_scope() as session:
        user = get_current_user(session)
    if not user:
        st.error("세션이 만료되었습니다. 다시 로그인해주세요.")
        st.session_state.pop("user", None)
        rerun_app()
        return
    if user.role == "producer":
        render_producer_portal(user)
    else:
        render_admin_portal(user)


if __name__ == "__main__":
    main()
