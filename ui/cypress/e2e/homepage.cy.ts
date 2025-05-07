describe('Check if text exists on the page', () => {
  it('should find the text on the page', () => {
    cy.visit('http://localhost:3000'); // Replace with your dev server URL
    cy.contains('Guidellm').should('exist'); // Replace with the text you're checking for
  });
});
