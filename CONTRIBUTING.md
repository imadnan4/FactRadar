# Contributing to FactRadar

We love your input! We want to make contributing to FactRadar as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

## Pull Requests

Pull requests are the best way to propose changes to the codebase. We actively welcome your pull requests:

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Any contributions you make will be under the MIT Software License
In short, when you submit code changes, your submissions are understood to be under the same [MIT License](http://choosealicense.com/licenses/mit/) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using GitHub's [issue tracker](https://github.com/yourusername/factradar/issues)
We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/yourusername/factradar/issues/new); it's that easy!

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Development Setup

### Prerequisites
- Node.js (v18 or higher)
- Python (v3.8 or higher)
- Git

### Local Development

1. **Clone your fork**
   ```bash
   git clone https://github.com/yourusername/factradar.git
   cd factradar
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Install dependencies**
   ```bash
   # Frontend
   npm install --legacy-peer-deps
   
   # Backend
   cd backend
   pip install -r requirements.txt
   cd ..
   ```

4. **Set up environment**
   ```bash
   cp .env.example .env.local
   # Edit .env.local with your configuration
   ```

5. **Start development servers**
   ```bash
   # Terminal 1 - Backend
   cd backend && python app.py
   
   # Terminal 2 - Frontend
   npm run dev
   ```

### Code Style

- **Frontend**: Follow the existing TypeScript/React patterns
- **Backend**: Follow PEP 8 Python style guidelines
- **Commit messages**: Use conventional commits format
- **Code formatting**: Use Prettier for frontend, Black for Python

### Testing

- **Frontend**: Ensure the application builds and runs without errors
- **Backend**: Test API endpoints with sample data
- **Models**: Verify model predictions with test cases

## Development Guidelines

### Frontend Development
- Use TypeScript for type safety
- Follow React hooks patterns
- Use Tailwind CSS for styling
- Ensure responsive design
- Test across different browsers

### Backend Development
- Use Flask best practices
- Implement proper error handling
- Add logging for debugging
- Document API endpoints
- Ensure CORS is properly configured

### Machine Learning
- Document model performance metrics
- Include training data information
- Provide model versioning
- Add reproducibility notes

## Submitting Changes

### Before Submitting
1. Test your changes thoroughly
2. Update documentation if needed
3. Check for any merge conflicts
4. Ensure all tests pass

### Pull Request Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tested locally
- [ ] Added tests
- [ ] Updated documentation

## Screenshots (if applicable)
Add screenshots here

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

## Community

### Communication
- **Issues**: Use GitHub issues for bug reports and feature requests
- **Discussions**: Use GitHub discussions for questions and ideas
- **Code Review**: All PRs require review before merging

### Code of Conduct
Be respectful and inclusive. We follow the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).

## Questions?
Feel free to open an issue or reach out to the maintainers. We're here to help!

## Recognition
Contributors will be recognized in our README and release notes. Thank you for helping make FactRadar better!