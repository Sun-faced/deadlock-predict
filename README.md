# Deadlock-Match-Predictor

## INFO

This is an experiment of how good can ML algorithms can predict outcomes of ranked matches based on unranked data.

This experiments uses data from "Old Gods New Blood" patch which came out on January 23, 2026 and ended on October 5, 2026. 

## Goal

We want to show that algorithm can predict ranked matches outcomes in 65%+ of the time. In addition, we would like to test how good can it predict the time match will last. 

Success will indicate that this framework could be used for future match predictions in later smaller patches, by only modifying data collection algorithms.

## Tests 
We are planning of conducting 6 experiments/models:

| #   | Name | Size | Weights | Reason for testing |
| -------- | ------- | ------- | ------- | ------- | 
| 0| Baseline| None | None |  Baseline to check if we get any result|
| 1 |Linear Model| We add 12 heroes info |no hidden layers | 128 | How much heroes alone explain|
| 2 | Team Sum | We add info about players playing together + 12 integers

Each model will receive the same 63 integers input tensor:
- 12 integers are allocated for chosen heroes (Deadlock is 6x6 game)
- 12 integers are allocated to show each hero's party size (0 means solo, 6 means that hero is playing in team of 6)
- 24 integers to show experience of the player on hero before the game (1 integer to show how many games, 1 integer to show winrate percentage)
- 12 integers to show assigned lanes (first stage of the game called 'lane stage' is most of the time crucial towards game success)
- 3 integers to show pool flags (those flags include 'new_player_pool', 'low_pri_pool', 'is_high_skill_range_parties'). Their purpose is to distinguish this data because new players are playing completely differently compared to experienced.







### Training

The last input tensor consists of 63 integers for which

