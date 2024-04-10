import spacy
import spacy_transformers
import sys
import io
import fitz
from Parser.BaseParser import BaseResumeInterface
from spacy.matcher import Matcher
import re
from collections import defaultdict
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFSyntaxError


class EnglishResume(BaseResumeInterface):

    def extract_education(self):
        regex = r"(?i)(education(?:al)?\s*(?:qualifications?|background)?|qualifications?|background)\s*:?\s*(.*?)(?=experience|certifications?|skills|$)"
        edu=[]

        matches = re.finditer(regex, self.resume_text, re.MULTILINE)

        for matchNum, match in enumerate(matches, start=1):

            print("Match {matchNum} was found at {start}-{end}: {match}".format(matchNum=matchNum, start=match.start(),
                                                                                end=match.end(), match=match.group()))

            for groupNum in range(0, len(match.groups())):
                groupNum = groupNum + 1

                print("Group {groupNum} found at {start}-{end}: {group}".format(groupNum=groupNum,
                                                                                start=match.start(groupNum),
                                                                                end=match.end(groupNum),
                                                                                group=match.group(groupNum)))
                edu.append(match.group(groupNum))

        return edu


    def extract_experience(self):

        wordnet_lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))

        # word tokenization
        word_tokens = nltk.word_tokenize(self.resume_text)

        # remove stop words and lemmatize
        filtered_sentence = [
            w for w in word_tokens if w not
                                      in stop_words and wordnet_lemmatizer.lemmatize(w)
                                      not in stop_words
        ]
        sent = nltk.pos_tag(filtered_sentence)
        print(filtered_sentence)
        # parse regex
        cp = nltk.RegexpParser('P: {<NNP>+}')
        cs = cp.parse(sent)

        # for i in cs.subtrees(filter=lambda x: x.label() == 'P'):
        #     print(i)

        test = []

        for vp in list(
                cs.subtrees(filter=lambda x: x.label() == 'P')
        ):
            # print("----")
            # print(vp.leaves())
            test.append(" ".join([
                i[0] for i in vp.leaves()
                if len(vp.leaves()) >= 1])
            )

        print(test)
        # Search the word 'experience' in the chunk and
        # then print out the text after it
        x = [
            x[x.lower().index('experience') + 10:]
            for i, x in enumerate(test)
            if x and 'experience' in x.lower()
        ]
        return x

    def extract_skillset(self):
        tokens = [token.text for token in self.nlp_text if not token.is_stop]
        skillset=[]
        for token in tokens:
            if token.lower() in self.skill_list:
                skillset.append(token)

        for token in self.nlp_text.noun_chunks:
            token = token.text.lower().strip()
            if token in self.skill_list:
                skillset.append(token)

        return [i.capitalize() for i in set([i.lower() for i in skillset])]

    def extract_phone(self):
        mob_num_regex = r'''(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)
                                [-\.\s]*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})'''
        phone = re.findall(re.compile(mob_num_regex), self.resume_text)
        if phone:
            number = ''.join(phone[0])
            return number

    def extract_email(self):

        email = re.findall(r"([^@|\s]+@[^@]+\.[^@|\s]+)", self.resume_text)
        if email:
            try:
                return email[0].split()[0].strip(';')
            except IndexError:
                return None

    def extract_website_urls(self):
        pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(pattern, self.resume_text)
        return urls

    def extract_name(self):

        pattern = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]

        self.matcher.add('NAME',[pattern])

        matches = self.matcher(self.nlp_text)

        for _, start, end in matches:
            span = self.nlp_text[start:end]
            if 'name' not in span.text.lower():
                return span.text


    def spacy_ent(self):
        # print("name by spacy")
        # ocself.nlp(self.resume_text)
        print("Running the custom Spacy model")
        temp_dct={}
        temp_dct['education'] = {
            'college': [],
            'degree': []
        }
        temp_dct['personal'] = {}
        temp_dct['work_exp'] = []

        doc=self.custom_nlp(self.resume_text,disable=['parser',"tagger","textcat"])

        for ent in doc.ents:
            if ent.label_ == 'DEGREE':
                temp_dct['education']['degree'].append(ent.text)
            elif ent.label_ == 'UNIVERSITY':
                temp_dct['education']['college'].append(ent.text)
            elif ent.label_ == 'NAME':
                temp_dct['personal']['name'] = ent.text
            elif ent.label_ == 'LOCATION':
                temp_dct['personal']['location'] = ent.text
            elif ent.label_ == 'WORKED AS':
                temp_dct['work_exp'].append({'designation': ent.text})
        return temp_dct


    def __init__(self,nlp,custom_nlp,skill_list,resume_text):

        self.nlp =nlp
        self.skill_list=skill_list
        self.custom_nlp=custom_nlp
        self.resume_text=resume_text
        self.nlp_text = self.nlp(self.resume_text)
        self.matcher = Matcher(self.nlp.vocab)

    def parse(self):
        dct={}
        dct['contact_info']={
            "email":self.extract_email(),
            "phone":self.extract_phone(),
            "websites":self.extract_website_urls()
        }
        dct['skills']=self.extract_skillset()

        dct['education']={
            'college':[],
            'degree':[]
        }
        dct['personal']={}
        dct['work_exp']=[]
        dct1=self.spacy_ent()
        return  {**dct,**dct1}
