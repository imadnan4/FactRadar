# Changelog

All notable changes to the FactRadar project will be documented in this file.

## [1.2.0] - 2025-06-23

### Added
- Multiple model selection feature
  - Added support for choosing between Gradient Boosting, LSTM, and CNN models
  - Implemented model dropdown in the UI next to the Analyze button
  - Created backend endpoints to support model switching
- Glass-morphism UI enhancements
  - Applied backdrop blur effects to result containers
  - Improved visual consistency across the application

### Changed
- Removed navigation bar for a cleaner interface
- Improved result display with glass-effect styling while maintaining color indicators
- Reorganized UI elements for better user experience
- Updated model selection UI to be more accessible

### Fixed
- Fixed alignment issues with UI elements
- Improved error handling for model loading
- Enhanced responsive design for mobile devices

## [1.1.0] - 2025-06-21

### Added
- AI Cross Verification using OpenRouter Mistral
- Detailed reasoning display for verification results
- Support for processing longer text inputs
- Performance metrics display for model evaluation

### Changed
- Improved text preprocessing pipeline
- Enhanced feature extraction for better accuracy
- Updated UI with new aurora background effects
- Optimized model loading for faster startup

### Fixed
- Fixed issues with text encoding in non-English content
- Resolved memory leaks in model prediction
- Fixed mobile responsiveness issues

## [1.0.0] - 2025-06-18

### Added
- Initial release of FactRadar
- Basic fake news detection using Gradient Boosting model
- Text input and analysis functionality
- Result display with confidence scores
- Sample text feature for demonstration
- Responsive design for desktop and mobile