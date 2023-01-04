from abc import abstractmethod


class AbstractRecordingsScource:
    @abstractmethod
    def get_recordings(self) -> dict[str, str]:
        """returns dict with recording name as key and reference to download recording as value"""
        pass

    @abstractmethod
    def upload_recording(self, name: str, recording):
        pass
