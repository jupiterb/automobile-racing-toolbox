from abc import abstractmethod


class AbstractRecordingsScource:
    @abstractmethod
    def get_recordings(self) -> dict[str, str]:
        """returns dict with recording name as key and url to download recording as value"""
        pass
