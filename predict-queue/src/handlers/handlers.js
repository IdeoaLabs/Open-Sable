/* Intent handlers */
module.exports = {
  'AMAZON.HelpIntent': function () {
    let speechOutput = '';
    speechOutput += 'I should contain help for your skill.';
    this.emit(':ask', speechOutput, speechOutput);
  },
  'AMAZON.StopIntent': function () {
    const speechOutput = 'Goodbye';
    this.emit(':tell', speechOutput);
  },

  'AMAZON.CancelIntent': function () {
    const speechOutput = 'Goodbye';
    this.emit(':tell', speechOutput);
  },
  LaunchRequest: function () {
    const firstAsk = 'Welcome to a sample skill created by create-alexa-skill. What would you like to do?';
    const noResponsePrompt = 'What would you like to do?';
    this.emit(':ask', firstAsk, noResponsePrompt);
  },
  Unhandled: function () {
    this.emit(':ask', 'I am not sure what you meant, say help me if you do not know what to say.', 'Say help me for help.');
  },
};
