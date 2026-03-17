/* eslint-env mocha */

const assert = require('assert');

const { whoIsNumberOne } = require('../src/index');

describe('Sample', () => {
  describe('#whoIsNumberOne()', () => {
    it('we should be number one', () => {
      assert.equal('We are number 1', whoIsNumberOne());
    });
  });
});
