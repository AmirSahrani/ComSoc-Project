## Computational Social Choice - Project

This project aims to investage distortion. Given that voters have utilities which they assign to each candidate,
distortion is the ratio of the candidate that would be selected by a voting rule after converting the utilities of the players to
(strict) linear orders of the candidate the would maximize the social welfare when accounting for the utilities.

In this repository we run simulations on different voting rules, using various notions of social welfare.
|Notion| Interpretation|
|---|---|
| Utilitarian | Maximize the total utility for all player |
| Rawlsian | Maximize the utility of the least happy player|
| Nietzschean | Maximize the the utility of the most happy player|

## Running the code
---
This project uses `uv` for its package management, but a `requirements.txt` file is also provided.

To run this projuect using `uv`:
```{bash}
uv run src/main.py
```
To run this in your own environment, first install the requiments:
```{bash}
pip install -r requirements.txt
```

Then simply run:
```{bash}
python src/main.py
```

## References
---
...
