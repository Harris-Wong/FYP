const fs = require('fs');
const { spawn } = require('child_process');
const cryptos = require("../models/cryptos");

const cryptoPairs = {
  "BTC-USD": "Bitcoin USD",
  "ADA-USD": "Cardano USD",
  "BCH-USD": "Bitcoin Cash USD",
  "DOGE-USD": "Dogecoin USD",
  "ETH-USD": "Ethereum USD",
  "FTT-USD": "FTX Token USD",
  "LINK-USD": "Chainlink USD",
  "LTC-USD": "Litecoin USD",
  "OKB-USD": "OKB USD",
  "SOL-USD": "Solana USD"
};

const getToday = () => {
  const today = new Date();
  const day = today.getDate().toString()
  const month = (today.getMonth() + 1).toString()
  const year = today.getFullYear().toString();
  const formattedDate = `${day}/${month}/${year}`;
  return formattedDate;
}

const index = async (req, res) => {
  if (!req.session.crypto) {
    // Default value for display: Bitcoin
    crypto = await cryptos.getCrypto("BTC-USD");
    req.session.crypto = crypto;
  }

  res.render("index", { crypto: req.session.crypto, cryptoPairs });
};

// Handle the get stock request
const getCryptoInfo = async (req, res) => {
  const code = req.body.cryptos;
  crypto = await cryptos.getCrypto(code);
  req.session.crypto = crypto;

  res.redirect("/");
};

// Handle the user's news input
const updateNewsInput = async (req, res) => {
  const newsInput = req.body.newsInput;
  const today = getToday();
  // Handle News Input
  const newsObject = `${newsInput}, ${today}\n`;

  fs.appendFile('./data/news/input_news.csv', newsObject, (err) => {
    if (err) throw err;
  });

  try {
    await runProcess('python3', ['Data_Scrap_Prediction_3.py']);
    console.log('News Prediction 1/2 completed');

    await runProcess('python3', ['Data_Scrap_Prediction_4.py']);
    console.log('News Prediction 2/2 completed');

  } catch (error) {
    console.error(error);
  }
  
  res.redirect("/");
};

function runProcess(command, args) {
  return new Promise((resolve, reject) => {
    const process = spawn(command, args);

    process.stdout.on('data', (data) => {
      console.log(`stdout: ${data}`);
    });

    process.stderr.on('data', (data) => {
      console.error(`stderr: ${data}`);
    });

    process.on('close', (code) => {
      if (code !== 0) {
        reject(`Process exited with code ${code}`);
      } else {
        resolve();
      }
    });
  });
}

module.exports = { index, getCryptoInfo, updateNewsInput};
