from sqlalchemy import Column, ForeignKey, Integer, String, Float
from sqlalchemy.orm import relationship
from sqlalchemy import Sequence
from db import Base


class Parser_Record(Base):
    __tablename__ = "parser_record"

    job_pk = Column(Integer,Sequence("parser_id_seq"), primary_key=True, index=True)
    remark = Column(String(50), nullable=False)
    status = Column(String(50), nullable=False)
    resume_pk = Column(Integer)
    candidate_pk=Column(Integer)
    language_code=Column(String(50), nullable=False)
    language_confidence= Column(Float)
    def __repr__(self):
        return 'ParserModel(remark=%s, status=%s,language_code=%s)' % (self.remark, self.status, self.language_code)


