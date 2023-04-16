const yahooFinance = require('yahoo-finance2').default;

async function getCrypto(code) {
    const summary = await yahooFinance.quoteSummary(code);
    const prices = await yahooFinance.historical(code,
        { period1: "2014-01-01" });

    prices.forEach((row, index) => {
        if (index == 0) {
            prices[index].actual = null;
        } else {
            prices[index].actual = (row.close - prices[index-1].close) >= 0;
        }
    });

    return { summary, prices };
}

module.exports = { getCrypto };
