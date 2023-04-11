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

  // Handle News Input
  const jsonNewsInput = JSON.parse(`{"NewsInput": "${newsInput}"}`);
  console.log(jsonNewsInput)

  res.redirect("/");
};

module.exports = { index, getCryptoInfo, updateNewsInput};
