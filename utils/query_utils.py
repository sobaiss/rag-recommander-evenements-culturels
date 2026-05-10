import datetime
import re


def expand_temporal_query(query: str, today: datetime.date | None = None) -> str:
    """Augmente la requête avec les dates ISO réelles des expressions temporelles françaises.

    Exemple :
        "événements semaine prochaine" (le 2026-05-10)
        → "événements semaine prochaine [période: du 2026-05-11 au 2026-05-17]"

    Cela permet à l'embedding FAISS de mieux matcher les chunks qui contiennent
    des dates explicites comme "DATES : du 2026-05-11 au 2026-05-15".
    """
    if today is None:
        today = datetime.date.today()

    q = query.lower()
    hint: str | None = None

    if re.search(r"\baujourd['\s]?hui\b", q):
        hint = today.isoformat()

    elif re.search(r"\bdemain\b", q):
        hint = (today + datetime.timedelta(days=1)).isoformat()

    elif re.search(r"\bce\s+week[\s-]?end\b|\bweekend\b|\bce\s+we\b", q):
        days_to_sat = (5 - today.weekday()) % 7 or 7
        sat = today + datetime.timedelta(days=days_to_sat)
        hint = f"du {sat.isoformat()} au {(sat + datetime.timedelta(days=1)).isoformat()}"

    elif re.search(r"\bsemaine\s+prochaine\b|\bla\s+semaine\s+prochaine\b", q):
        days_to_mon = (7 - today.weekday()) % 7 or 7
        mon = today + datetime.timedelta(days=days_to_mon)
        sun = mon + datetime.timedelta(days=6)
        hint = f"du {mon.isoformat()} au {sun.isoformat()}"

    elif re.search(r"\bcette\s+semaine\b|\bcette\s+semaine\b", q):
        mon = today - datetime.timedelta(days=today.weekday())
        sun = mon + datetime.timedelta(days=6)
        hint = f"du {mon.isoformat()} au {sun.isoformat()}"

    elif re.search(r"\bce\s+mois\b|\bce\s+mois-ci\b", q):
        first = today.replace(day=1)
        next_month = (today.replace(day=28) + datetime.timedelta(days=4)).replace(day=1)
        last = next_month - datetime.timedelta(days=1)
        hint = f"du {first.isoformat()} au {last.isoformat()}"

    elif re.search(r"\bmois\s+prochain\b", q):
        first = (today.replace(day=28) + datetime.timedelta(days=4)).replace(day=1)
        last = (first.replace(day=28) + datetime.timedelta(days=4)).replace(day=1) - datetime.timedelta(days=1)
        hint = f"du {first.isoformat()} au {last.isoformat()}"

    if hint:
        return f"{query} [période: {hint}]"
    return query
