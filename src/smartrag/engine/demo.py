"""A small demo corpus: company rules and an expense manual (Japanese), a product README
(English), across two sources. Used by `smartrag serve --demo` and the API tests."""

from __future__ import annotations

from datetime import datetime, timezone

from smartrag.engine.memory import Chunk, Document, InMemoryEngine, Source

_SYNCED = datetime(2026, 11, 8, 3, 0, tzinfo=timezone.utc)


def demo_engine() -> InMemoryEngine:
    leave = ("第5章 休暇", "第22条 年次有給休暇")
    return InMemoryEngine(
        sources=[
            Source("nas-kitei", "社内規程", "smb", _SYNCED),
            Source("s3-manuals", "製品マニュアル", "s3", _SYNCED),
        ],
        documents=[
            Document(
                source_id="nas-kitei",
                path="就業規則/就業規則.pdf",
                title="就業規則",
                file_type="pdf",
                version=3,
                modified_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
                chunks=[
                    Chunk("年次有給休暇は、入社後6か月間継続勤務し、所定労働日の8割以上出勤した"
                          "従業員に10日付与する。", leave, 14),
                    Chunk("年次有給休暇は、翌年度に限り最大20日まで繰り越すことができる。"
                          "繰り越した休暇は、当年度に付与された休暇より先に取得したものとみなす。",
                          leave, 14),
                    Chunk("従業員が結婚するときは5日、配偶者が出産するときは3日の特別休暇を与える。",
                          ("第5章 休暇", "第23条 特別休暇"), 15),
                    Chunk("従業員は、会社の許可を得た場合に限り、副業を行うことができる。",
                          ("第6章 服務", "第30条 副業"), 18),
                ],
            ),
            Document(
                source_id="nas-kitei",
                path="経理/経費精算マニュアル.docx",
                title="経費精算マニュアル",
                file_type="docx",
                version=1,
                modified_at=datetime(2025, 10, 15, tzinfo=timezone.utc),
                chunks=[
                    Chunk("経費は発生から30日以内に経費精算システムで申請する。"
                          "領収書の画像を必ず添付すること。", ("1. 申請の流れ",)),
                    Chunk("出張の交通費は最も経済的な経路で計算する。"
                          "新幹線のグリーン車は部長以上に限り利用できる。", ("2. 交通費",)),
                ],
            ),
            Document(
                source_id="s3-manuals",
                path="gotobox/README.md",
                title="GotoBox Quick Start",
                file_type="md",
                version=2,
                modified_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
                chunks=[
                    Chunk("GotoBox connects to your file server and indexes documents "
                          "every night at 3:00.", ("Quick Start",)),
                    Chunk("If indexing fails, check that the service account can read the "
                          "shared folder.", ("Quick Start", "Troubleshooting")),
                ],
            ),
        ],
    )
