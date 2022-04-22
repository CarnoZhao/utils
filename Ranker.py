from bs4 import BeautifulSoup
import requests
from rich.console import Console
from rich.table import Table
import pandas as pd

urlA = "https://www.heywhale.com/v2/api/tasks/62353f220cfed900174d7ef1/leaderboard?template=true"
urlB = "https://www.heywhale.com/v2/api/tasks/625f69d2697f5d0018062ef9/leaderboard?template=true"
textA = requests.get(urlA).text
textB = requests.get(urlB).text

def get_score(text):
    soup = BeautifulSoup(text, "html.parser")
    scores = {}
    for team in  soup.findAll("tr", {"data-team-id": True}):
        team_name = team.findAll("td", {"class": lambda x: x == "team-cell"})[0].text.strip()
        team_name = team_name.replace("的团队", "")
        score = float(team.findAll("td", {"class": lambda x: x == "submit-score"})[0].text.strip().split("\n")[0])
        scores[team_name] = score
    return scores

A, B = get_score(textA), get_score(textB)

dfA = pd.DataFrame({"team_name": A.keys(), "score_A": A.values()})
dfA = dfA.sort_values("score_A", ascending = False).reset_index(drop = True)
dfA["rank_A"] = dfA.index + 1
dfB = pd.DataFrame({"team_name": B.keys(), "score_B": B.values()})
dfB = dfB.sort_values("score_B", ascending = False).reset_index(drop = True)
dfB["rank_B"] = dfB.index + 1
df = pd.merge(dfA, dfB, how = "outer")
df.rank_A = df.rank_A.fillna(df.rank_A.max() + 1).astype(int)
df.rank_B = df.rank_B.fillna(df.rank_B.max() + 1).astype(int)
df.score_A = df.score_A.fillna(0)
df.score_B = df.score_B.fillna(0)
df = df.sort_values(["rank_B", "rank_A"])
df["score_change"] = df.score_B - df.score_A
df["rank_change"] = ["+" + str(_) if _ > 0 else str(_) for _ in (df.rank_A - df.rank_B) * (df.score_B != 0)]


table = Table(row_styles = ["dim", ""])
colors = ["red", "green", "blue", "green", "blue", "green", "blue"]
for i, col in enumerate(df.columns):
    table.add_column(col, style = colors[i])


for _, row in df.iterrows():
    table.add_row(*[x if isinstance(x, str) else str(x) if isinstance(x, int) else f"{x:.4f}" for x in row])


console = Console()
console.print(table)
