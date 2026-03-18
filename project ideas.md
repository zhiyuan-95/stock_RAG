**stage 1 (prepare data)**

~~1. get record financial indicators 8 year annual, 12 quarter~~

~~2. saved in sql\_db, and vector store, data in sql\_db will be updated first, and vector store will be updated based on the change in sql\_db~~



3\. get upto date market environment

(fed interest rate, GDP, inflation rate(CPI), unemployment rate, ADP National Employment Report, US BLS Employment Situation Report, PMI(manuf, service))



4\. get current price of raw materials, gold, crude oil(gas price), btc, (including 20, 50, 100, MA)

5\. get current price and track moving average of all major indicators

6\. get industrial/sector average of all different indicators, including  VIX, bond yeild

7\. for each stock, get a place to store its major competitors



**stage 2 (news collection)**

8\. get major headlines every week, US, international, and store it in vector storage(probably I am gonna need a ranking system to rate the influence of on the stock market)

9\. collect corresponding news for each stocks...

how often should I update them?

do I actually need all the news for all the stocks in the market?



**stage 3 (smarter)**

10\. get better prompt to get better response

11\. fine tuning the base model with research paper



### **questions**

1. does the language model can actually understand those time series data of those financial indicators in doc and give a meaningful response?
2. verify if the answer from the llm actually match with actually data

