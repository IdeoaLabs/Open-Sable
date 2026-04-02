const Alexa = require('alexa-sdk');
const handlers = require('./handlers/handlers');

/* Basic config check */
if (!process.env.appId) {
  console.error('Please ensure you have set the appId environment variable.');
  process.exit(1);
}

/* Basic function to show Mocha tests, you can remove this */
function whoIsNumberOne() {
  return 'We are number 1';
}

console.log(whoIsNumberOne());

exports.whoIsNumberOne = whoIsNumberOne;
/* End basic function section */

/* Alexa Handlers are pulled from /handlers and given here */
exports.handler = (event, context) => {
  const alexa = Alexa.handler(event, context);
  alexa.appId = process.env.appId;
  alexa.registerHandlers(handlers);
  alexa.execute();
};