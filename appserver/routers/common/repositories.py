from repository import GuardRepository, InMemoryRepository
from schemas import Game, Training


class Repositories:

    games: InMemoryRepository[str, Game] = InMemoryRepository()

    trainings: GuardRepository[str, tuple[str, str], Training]= GuardRepository(
        InMemoryRepository[tuple[str, str], Training](),
        games,
        lambda game_training_id: game_training_id[0]
)