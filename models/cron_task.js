const { spawn } = require('child_process');
const { apiFetch } = require('./data_fetch.js');

async function predictionTask() {
  try {
    // Run the JavaScript file
    // await runProcess('node', ['-e', `${apiFetch.toString()}; apiFetch()`]);
    // console.log('Data Fetch completed');

    // // Run the Python files in order
    // await runProcess('python', ['Data_Scrap_Prediction_1.py']);
    // console.log('Data Scrap Prediction 1/4 completed');

    // await runProcess('python', ['Data_Scrap_Prediction_2.py']);
    // console.log('Data Scrap Prediction 2/4 completed');

    // await runProcess('python', ['Data_Scrap_Prediction_3.py']);
    // console.log('Data Scrap Prediction 3/4 completed');

    await runProcess('python', ['Data_Scrap_Prediction_4.py']);
    console.log('Data Scrap Prediction 4/4 completed');
    console.log('All Prediction completed');

  } catch (error) {
    console.error(error);
  }
}

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

module.exports = { predictionTask };
