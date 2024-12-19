## Computational Social Choice - Project

This project aims to investage distortion. Given that voters have utilities which they assign to each candidate,
distortion is the ratio of the candidate that would be selected by a voting rule after converting the utilities of the players to
(strict) linear orders of the candidate the would maximize the social welfare when accounting for the utilities.

In this repository we run simulations on different voting rules, using various notions of social welfare.
|Notion| Interpretation|
|---|---|
| Utilitarian | Maximize the total utility for all player |
| Nashian | Maximize the product of utilities for all players |
| Rawlsian | Maximize the utility of the least happy player|
| Nietzschean | Maximize the the utility of the most happy player|

## Running the code
This project uses `uv` for its package management, but a `requirements.txt` file is also provided.

To run this simulations using `uv`:
```{bash}
uv run src/main.py
```
To generate the statistics, figures and tables you can use the same command but with `src/stats.py`, `src/plotting.py`, `src/tablemaking.py` respectively.

To run this in your own environment, first install the requiments:
```{bash}
pip install -r requirements.txt
```

Then simply run:
```{bash}
python src/main.py
```

Figures will be shown and saved in `/figures/`, the results of the simulations and statistics will be saved under `/results/`. The `data` folder contains the data from the Sushi dataset [1]

## References
```
@Inbook{Kamishima2011,
author="Kamishima, Toshihiro
and Kazawa, Hideto
and Akaho, Shotaro",
title="A Survey and Empirical Comparison of Object Ranking Methods",
bookTitle="Preference Learning",
year="2011",
publisher="Springer Berlin Heidelberg",
address="Berlin, Heidelberg",
pages="181--201",
abstract="Ordered lists of objects are widely used as representational forms. Such ordered objects include Web search results or bestseller lists. In spite of their importance, methods of processing orders have received little attention. However, research concerning orders has recently become common; in particular, researchers have developed various methods for the task of Object Ranking to acquire functions for object sorting from example orders. Here, we give a unified view of these methods and compare their merits and demerits.",
isbn="978-3-642-14125-6",
doi="10.1007/978-3-642-14125-6_9",
url="https://doi.org/10.1007/978-3-642-14125-6_9"
}
```
