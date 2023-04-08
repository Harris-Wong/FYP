// https://www.highcharts.com/demo/stock
// https://www.highcharts.com/demo/stock/candlestick-and-volume

function createChart(data, name) {

    // split the data set into ohlc and volume
    var ohlc = [],
    volume = [],
    dataLength = data.length,
    // set the allowed units for data grouping
    groupingUnits = [[
      'week',             // unit name
      [1]               // allowed multiples
    ], [
      'month',
      [1, 2, 3, 4, 6]
    ]],

    i = 0;

  for (i; i < dataLength; i += 1) {
    // ohlc.push([
    //   data[i][0], // the date
    //   data[i][1], // open
    //   data[i][2], // high
    //   data[i][3], // low
    //   data[i][4] // close
    // ]);
    //
    // volume.push([
    //   data[i][0], // the date
    //   data[i][5] // the volume
    // ]);
    ohlc.push([
      Date.parse(data[i].date), // the date
      data[i].open, // open
      data[i].high, // high
      data[i].low, // low
      data[i].adjClose // close
    ]);

    volume.push([
      Date.parse(data[i].date), // the date
      data[i].volume // the volume
    ]);
  }

  // create the chart
  Highcharts.stockChart('chart', {

    rangeSelector: {
      selected: 1
    },

    title: {
      text:  'Candlestick Chart'
    },

    yAxis: [{
      labels: {
        align: 'right',
        x: -3
      },
      title: {
        text: 'OHLC'
      },
      height: '60%',
      lineWidth: 2,
      resize: {
        enabled: true
      }
    }, {
      labels: {
        align: 'right',
        x: -3
      },
      title: {
        text: 'Volume'
      },
      top: '65%',
      height: '35%',
      offset: 0,
      lineWidth: 2
    }],

    tooltip: {
      split: true
    },

    series: [{
      type: 'candlestick',
      name: name,
      data: ohlc,
      dataGrouping: {
        units: groupingUnits
      }
    }, {
      type: 'column',
      name: 'Volume',
      data: volume,
      yAxis: 1,
      dataGrouping: {
        units: groupingUnits
      }
    }]
  });
};
