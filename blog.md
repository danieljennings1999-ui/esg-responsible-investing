# Investing Green: What the Data Actually Shows

*September 2026*

London just had its hottest and driest summer on record. As someone 
working in ETF and index data at Ultumus, I kept asking the same 
question my clients ask: does going green cost you money as an investor?

I spent three months building an answer using real data.

## What I Did
I pulled environmental scores for 1,496 US companies from Refinitiv, 
the same data used by institutional asset managers, and matched them 
to real stock price histories from 2022 to 2025. I split companies into 
a Green portfolio (top 30% environmental scorers) and a Brown portfolio 
(bottom 30%) and ran both through a rigorous backtest.

## What I Found
The headline result: Green returned 21.1%, Brown returned 24.7%, and 
the unconstrained market returned 20.3% over 20 months. Statistically 
those are indistinguishable, there is no financial penalty for 
investing responsibly.

But the more interesting finding was hidden underneath. When I split 
companies by market cap, better environmental scores consistently 
predicted better returns within every size group. The aggregate result 
was misleading, a classic case of Simpson's Paradox.

## Why It Matters
For ETF issuers and asset managers facing growing ESG compliance 
pressure, this analysis provides quantitative evidence that 
environmental screening is financially viable. You do not have to 
choose between doing good and doing well.

## Tools Used
Python, PyPortfolioOpt, Refinitiv, yfinance, Matplotlib, scipy

## Read More
Full report and code available in this repository.
