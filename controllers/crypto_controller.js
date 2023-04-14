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

let newsArray = [];

const getToday = () => {
  const today = new Date();
  const year = today.getUTCFullYear().toString();
  const month = (today.getUTCMonth() + 1).toString().padStart(2, '0');
  const day = today.getUTCDate().toString().padStart(2, '0');
  const formattedDate = `${day}/${month}/${year}`;
  return formattedDate;
};

const index = async (req, res) => {
  if (!req.session.crypto) {
    // Default value for display: Bitcoin
    const crypto = await cryptos.getCrypto("BTC-USD");
    req.session.crypto = crypto;
  }

  // Get Signal Data
  let signalValue;
  fs.readFile('./data/signals.json', 'utf8', (err, data) => {
    if (err) {
      console.log(err);
      return;
    }
    const signalData = JSON.parse(data);
    for (const key in signalData) {
      if (key == (req.session.crypto.summary.price.symbol).substring(0, 3)) {
        signalValue = signalData[key];
      }
    }
  });

  res.render("index", { crypto: req.session.crypto, cryptoPairs, signalValue, newsArray });
};

// Handle the get stock request
const getCryptoInfo = async (req, res) => {
  const code = req.body.cryptos;
  const crypto = await cryptos.getCrypto(code);
  req.session.crypto = crypto;

  res.redirect("/");
};

// Handle the user's news input
const updateNewsInput = async (req, res) => {
  if (req.body.predict_button) {
    // Perform operation for Predict button
    const newsInput = req.body.newsInput;

    if (newsInput && newsInput.trim() !== '') {
      // check for empty or space input
      const today = getToday();
      // Handle News Input
      const newsObject = `${newsInput}, ${today}\n`;
     
      fs.appendFile('./data/news/input_news.csv', newsObject, (err) => {
        if (err) throw err;
  
        const newsFile = fs.readFileSync('./data/news/input_news.csv', 'utf-8');
        const rows = newsFile.split('\n');
        
        // update frontend news
        newsArray = [];
        for (let i = 1; i < rows.length-1; i++) {
          const news = rows[i].split(',')[0];
          newsArray.push(news);
        }
      })
    }
      try {
        await runProcess('python', ['Data_Scrap_Prediction_3.py']);
        console.log('News Prediction 1/2 completed');
    
        await runProcess('python', ['Data_Scrap_Prediction_4.py']);
        console.log('News Prediction 2/2 completed');
      } catch (error) {
        console.error(error);
      }
  } else if (req.body.clear_button) {
    // Perform operation for Clean button
    const csvNewsHeader = 'input,date\n';
    fs.writeFile('data/news/input_news.csv', csvNewsHeader, (err) => {
      if (err) throw err;
    });

    try {
      await runProcess('python', ['Data_Scrap_Prediction_3.py']);
      console.log('News Prediction 1/2 completed');
      
      await runProcess('python', ['Data_Scrap_Prediction_4.py']);
      console.log('News Prediction 2/2 completed');
    } catch (error) {
      console.error(error);
    }
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
