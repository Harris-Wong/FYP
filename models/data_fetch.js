function apiFetch() {
  const { investing } = require('investing-com-api');
  const yahooFinance = require('yahoo-finance2').default;
  const axios = require('axios');
  const fs = require('fs');
  const url = require('url');


  function paramsToQuery(params) {
    return Object.keys(params)
            .map(k => encodeURIComponent(k) + '=' + encodeURIComponent(params[k]))
            .join('&');
  }

  function dateRange(startDate, endDate) {
    const dateArray = [];
    let currentDate = new Date(startDate);

    while (currentDate <= new Date(endDate)) {
      dateArray.push(new Date(currentDate).toJSON().slice(0, 10));
      // Use UTC date to prevent problems with time zones and DST
      currentDate.setUTCDate(currentDate.getUTCDate() + 1);
    }

    return dateArray;
  }

  function addDays(date, days) {
    date.setDate(date.getDate() + days);
    return date;
  }

  function diffDays(date1, date2) {
    const diffTime = Math.abs(date2 - date1);
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    return diffDays;
  }

  function exportToFile(data, filename) {
    let jsonData = JSON.stringify(data);
    fs.writeFile(filename+'.txt', jsonData, function(err) {
     if (err) {
        console.log(err);
      }
    });
  }

  const crypto_list = {
    'BTC-USD': 'BTC',
    'ADA-USD': 'ADA',
    'BCH-USD': 'BCH',
    'BNB-USD': 'BNB',
    'DOGE-USD': 'DOGE',
    'ETH-USD': 'ETH',
    'FTT-USD': 'FTT',
    'LINK-USD': 'LINK',
    'OKB-USD': 'OKB',
    'SOL-USD': 'SOL',
  };

  const startDate = '2014-01-01';
  const endDate = new Date().toJSON().slice(0, 10);
  const dates = dateRange(startDate, endDate);
  var data = [];
  dates.forEach(date => data.push({ Date: date }));

  async function fetchyFinance(yFinanceData) {
    const responses = await Promise.all(
      Object.keys(yFinanceData).map(async (item) => {
        return await yahooFinance.historical(item, { period1: "2014-01-01" });
      })
    );

    responses.forEach((response, responseIndex) => {
      let resIndex = 0;
      const resLength = response.length;

      data.forEach((row, rowIndex) => {
        if (resIndex < resLength) {
          if (row.Date != response[resIndex].date.toISOString().slice(0,10)) {
            if (Object.keys(yFinanceData)[responseIndex] in crypto_list) {
              data[rowIndex]['Open'] = null;
              data[rowIndex]['High'] = null;
              data[rowIndex]['Low'] = null;
              data[rowIndex]['Adj Close'] = null;
              data[rowIndex]['Volume'] = null;
            } else {
              data[rowIndex][Object.values(yFinanceData)[responseIndex]] = null;
            }
          } else {
            if (Object.keys(yFinanceData)[responseIndex] in crypto_list) {
              data[rowIndex]['Open'] = response[resIndex].open;
              data[rowIndex]['High'] = response[resIndex].high;
              data[rowIndex]['Low'] = response[resIndex].low;
              data[rowIndex]['Adj Close'] = response[resIndex].close;
              data[rowIndex]['Volume'] = response[resIndex].volume;

            } else {
              data[rowIndex][Object.values(yFinanceData)[responseIndex]] = response[resIndex].adjClose;
            }
            resIndex++;
          }
        } else {
          if (Object.keys(yFinanceData)[responseIndex] in crypto_list) {
            data[rowIndex]['Open'] = null;
            data[rowIndex]['High'] = null;
            data[rowIndex]['Low'] = null;
            data[rowIndex]['Adj Close'] = null;
            data[rowIndex]['Volume'] = null;
          } else {
            data[rowIndex][Object.values(yFinanceData)[responseIndex]] = null;
          }
        };
      });
    });
  }

  async function fetchMessari() {
    // Bitcoin - Ciculating Supply -> https://messari.io/api/docs#tag/Timeseries/operation/Get%20Market%20timeseries

    const assets = 'bitcoin';
    const metrics = {'sply.circ': 'Bitcoin_Circulating_Supply'}; // Metrics list -> https://data.messari.io/api/v1/assets/metrics
    const periods = [
      { 'start': startDate, 'end': '2018-12-31'},
      { 'start': '2019-01-01', 'end': endDate }
    ];
    const params = {
      'start': '',
      'end': '',
      'interval': '1d',
      'order': 'ascending',
      'format': 'json',
      'timestamp-format': 'rfc3339'
    };

    const requests = periods.map(period => {
      params.start = period.start;
      params.end = period.end;
      return axios.get(`https://data.messari.io/api/v1/assets/${assets}/metrics/${Object.keys(metrics)}/time-series?` + paramsToQuery(params))
    });

    let rowIndex = 0;
    result = await axios.all(requests).then(responses => {
      responses.forEach(response => {
        response = response.data.data.values;
        response.forEach(row => {
          // data[rowIndex]['date'] = row[0].slice(0,10),
          data[rowIndex][Object.values(metrics)] = row[1];
          rowIndex++;
        });
      });
    });
  }

  async function fetchNasdaq() {
    // Blockchain Data - Bitcoin -> https://data.nasdaq.com/tools/api

    const nasdaqData = {
      'AVBLS': 'Bitcoin_Block_Size', // Avg Block Size
      'CPTRA': 'Bitcoin_Transaction_Cost', // Cost Per Transaction
      'NTRAN': 'Bitcoin_Daily_Transaction', // Daily Transaction
      'HRATE': 'Bitcoin_Hash_Rate' // Hash Rate
    };
    const params = {'api_key': 'yhS2tz5irv2PxPZLsskh'};
    const requests = Object.keys(nasdaqData).map(data =>
      axios.get(`https://data.nasdaq.com/api/v3/datasets/BCHAIN/${data}.csv?` + paramsToQuery(params), { responseType: 'blob',})
    );

    result = await axios.all(requests).then(responses => {
      responses.forEach((response, responseIndex) => {
        responseArray = response.data.split('\n').slice(1, -1);

        responseArray
          .filter(row => new Date(row.split(",")[0]) >= new Date(startDate))
          .reverse()
          .forEach((row, rowIndex) => {
            // data[rowIndex]['date'] = row.split(",")[0],
            data[rowIndex][Object.values(nasdaqData)[responseIndex]] = row.split(",")[1];
        });
        responseIndex++;
      });
    });
  }

  async function fetchEtherscan() {
    // Blockchain Data - Ethereum -> https://etherscan.io/chart/hashrate?output=csv

    const etherscanData = {
      'gasprice': 'Ethereum_Gas_Price', // Avg Gas Price
      'blockreward': 'Ethereum_Block_Reward', // Block Reward
      'tx': 'Ethereum_Daily_Transaction', // Daily Transaction
      'gasused': 'Ethereum_Gas_Usage', // Gas Used
      'hashrate': 'Ethereum_Hash_Rate', // Hash Rate
    };
    // params = {
    //   'apikey': 'QHWZ85DP2Y4H828W1DGJK53C6K8K69UJPP'
    // }

    const requests = Object.keys(etherscanData).map(data =>
      axios.get(`https://etherscan.io/chart/${data}?output=csv`)
    );

    result = await axios.all(requests).then((responses) => {
      responses.forEach((response, responseIndex) => {
        responseArray = response.data.split('\n').slice(1, -1);

        let resIndex = 0;
        const resLength = responseArray.length;

        data.forEach((row, rowIndex) => {
          if (resIndex < resLength) {
            if (row.Date != new Date(responseArray[resIndex].replace(/"/g, '').trim().split(',')[0]).toISOString().slice(0,10)) {
              data[rowIndex][Object.values(etherscanData)[responseIndex]] = null;
            } else {
              // data[rowIndex].date = new Date(responseArray[resIndex].replace(/"/g, '').trim().split(',')[0]).toISOString().slice(0,10);
              data[rowIndex][Object.values(etherscanData)[responseIndex]] = responseArray[resIndex].replace(/"/g, '').trim().split(',')[2];
              resIndex++;
            }
          } else {
            data[rowIndex][Object.values(etherscanData)[responseIndex]] = null;
          };
        });
      });
    });
  }

  async function fetchInvesting() {
    // https://www.npmjs.com/package/investing-com-api

    const investingData = {
      // 'commodities/copper',
      // 'commodities/gold',
      // 'commodities/crude-oil',
      // 'commodities/silver',
      'rates-bonds/india-10-year-bond-yield': 'India_10YBond',
      'rates-bonds/india-2-year-bond-yield': 'India_2YBond',
      'rates-bonds/uk-10-year-bond-yield': 'UK_10YBond',
      'rates-bonds/uk-2-year-bond-yield': 'UK_2YBond',
      'rates-bonds/u.s.-10-year-bond-yield': 'US_10YBond',
      'rates-bonds/u.s.-2-year-bond-yield': 'US_2YBond',
    };

    const responses = await Promise.all(
      Object.keys(investingData).map(async (item) => {
        return await investing(item, 'MAX', 'P1D');
      })
    );

    responses.forEach((response, resIndex) => {
      response.forEach((row, index) => {
        response[index].date = new Date(row.date);
      });

      response = response.filter(row => {
        return parseInt(row.date.getFullYear()) >= parseInt(startDate.slice(0,4));
      });

      let responseIndex = 0;
      const responseLength = response.length;

      data.forEach((row, index) => {
        if (responseIndex < responseLength) {
          if (row.Date != response[responseIndex].date.toISOString().slice(0,10)) {
            data[index][Object.values(investingData)[resIndex]] = null;
          } else {
            data[index][Object.values(investingData)[resIndex]] = response[responseIndex].price_close;
            responseIndex++;
          }
        } else {
          data[index][investingData[resIndex]] = null;
        };
      });
    });
  }

  async function fetchCryptoCompare() {
    // https://min-api.cryptocompare.com/documentation?key=Historical&cat=dataExchangeHistoday

    const cryptoCompareData = [
      'Binance',
      'BinanceUS', // Binance.US
      'Bitstamp',
      'Coinbase',
      'FTX',
      'Gateio', // Gate.io
      'Gemini',
      'Kraken',
      'KuCoin',
      'MEXC',
    ];

    const toTsLimit = [
      {
        'toTs': Date.parse(addDays(new Date(startDate),2000))/1000,
        'limit': 2000
      },
      {
        'toTs': Date.parse(new Date(endDate))/1000,
        'limit': diffDays(addDays(new Date(startDate),2000), new Date(endDate))-1
      }
    ];

    const params = {
      'tsym': 'USD',
      'e': 'Coinbase',
      'aggregate': '1',
      'limit': '',
      'toTs': '',
      'api_key': 'caecba08445b0321841ccce69887c645309103385434b2210cbda484f5285a2c'
    }

    let requests = cryptoCompareData.map(data => {
      return toTsLimit.map(period => {
        params.e = data;
        params.toTs = period.toTs;
        params.limit = period.limit;
        return axios.get(`https://min-api.cryptocompare.com/data/exchange/histoday?` + paramsToQuery(params));
      });
    });
    requests = requests.flat();

    let dataIndex = 0;
    let rowIndex = 0;
    result = await axios.all(requests).then(responses => {
      responses.forEach((response, responseIndex) => {
        response = response.data.Data;
        dataIndex = Math.ceil((responseIndex+1.0)/2)-1;
        if (dataIndex != (responseIndex+1.0)/2-1) rowIndex = 0;

        response.forEach(row => {
          // data[rowIndex][cryptoCompareData[dataIndex]+'_Date'] = new Date(row.time * 1e3).toISOString().slice(0, 10);
          data[rowIndex][cryptoCompareData[dataIndex]+'_Exchange_Volume'] = row.volume;
          rowIndex++;
        });
      });
    });
  }

  const apiCalls = async () => {
    const Bitcoin = {
    // Bitcoin
      'BTC-USD': 'Crypto_'+'BTC'
    };

    const Stock = {
      'BTCM': 'STOCK_'+'BTCM',
      'HIVE': 'STOCK_'+'HIVE',
      'VYGVQ': 'STOCK_'+'VYGVQ',
      'AMZN': 'STOCK_'+'AMAZON',
      'AAPL': 'STOCK_'+'APPLE',
      'GOOGL': 'STOCK_'+'GOOGLE',
      'META': 'STOCK_'+'META',
      'MSFT': 'STOCK_'+'MICROSOFT',
      'NFLX': 'STOCK_'+'NETFLIX',
      'TSLA': 'STOCK_'+'TESLA',
      'MSTR': 'STOCK_'+'MSTR',
    };

    const Index = {
      '^DJI': 'Index_'+'DJI',
      '^FCHI': 'Index_'+'FCHI',
      '^GDAXI': 'Index_'+'GDAXI',
      '^GSPTSE': 'Index_'+'GSPTSE',
      '^IXIC': 'Index_'+'IXIC',
      '^N225': 'Index_'+'N225',
      '^VIX': 'Index_'+'VIX',
      '^GSPC': 'Index_'+'SP500',
    };

    const Commodities = {
      'ALI=F': 'Aluminium'+'_Futures',
      'HG=F': 'Copper'+'_Futures',
      'GC=F': 'Gold'+'_Futures',
      'CL=F': 'Crude Oil'+'_Futures',
      'SI=F': 'Silver'+'_Futures',
    };

    const ETF = {
      'BITO': 'BITO'+' ETF',
      'BITS': 'BITS'+' ETF',
      'BITW': 'BITW'+' ETF',
      'BLCN': 'BLCN'+' ETF',
      'BLOK': 'BLOK'+' ETF',
      'BTF': 'BTF'+' ETF',
      'CRPT': 'CRPT'+' ETF',
      'ETHE': 'ETHE'+' ETF',
      'GBTC': 'GBTC'+' ETF',
      'XBTF': 'XBTF'+' ETF',
    };

    const Crypto = {
      'LTC-USD': 'Crypto_'+'LTC',
    };

    // For all Data fetch
    try {
       await fetchyFinance(Stock);
       await fetchyFinance(Index);
       await fetchMessari();
       await fetchNasdaq();
       await fetchEtherscan();
       await fetchInvesting();
       await fetchyFinance(Commodities);
       await fetchyFinance(ETF);
       await fetchCryptoCompare();
       await fetchyFinance(Crypto);
     } catch (error) {
       console.error(error);
     };
    exportToFile(data, 'all_data');

    // For Crypto fetch
    for (const crypto in crypto_list) {
      data = [];
      dates.forEach(date => data.push({ Date: date }));

      try {
        await fetchyFinance({[crypto]: crypto_list[crypto]});
      } catch (error) {
        console.error(error);
      };

      exportToFile(data, `${crypto_list[crypto]}_data`);
    }
  }

  apiCalls();
}

module.exports = { apiFetch };
