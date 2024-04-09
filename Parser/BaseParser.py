from abc import ABC, abstractmethod

class BaseResumeInterface(ABC):
    @abstractmethod
    def extract_education(self):
        pass

    @abstractmethod
    def extract_experience(self):
        pass

    @abstractmethod
    def extract_skillset(self):
        pass

    @abstractmethod
    def extract_phone(self):
        pass

    @abstractmethod
    def extract_email(self):
        pass

    @abstractmethod
    def extract_website_urls(self):
        pass

    @abstractmethod
    def extract_name(self):
        pass