# FactRadar - AI-Powered Fake News Detection

FactRadar is an advanced AI-powered application designed to detect fake news and misinformation in text content. It uses multiple machine learning models and AI verification to provide accurate analysis of news articles, social media posts, and other text sources.

## 🏗️ Project Structure

```
FactRadar/
├── 📁 app/                          # Next.js app directory
│   ├── globals.css                 # Global styles
│   ├── layout.tsx                  # Root layout
│   ├── page.tsx                    # Main application page
│   └── page.tsx.backup             # Backup of original page
├── 📁 backend/                     # Python Flask backend
│   ├── app.py                      # Main Flask application
│   ├── app.py.backup               # Backup of original app.py
│   ├── test.py                     # Backend testing utilities
│   ├── requirements.txt            # Python dependencies
│   ├── 📁 models/                  # ML models directory
│   │   ├── best_model_gradient_boosting.pkl
│   │   ├── best_model_metadata.json
│   │   ├── cnn_model.h5
│   │   ├── lstm_model.h5
│   │   └── tfidf_vectorizer_full.pkl
│   └── 📁 old/                     # Legacy model backups
├── 📁 components/                  # React components
│   ├── FakeNewsDetector.tsx        # Main detection component
│   ├── theme-provider.tsx          # Theme management
│   ├── theme-switcher.tsx          # Theme toggle component
│   ├── theme-toggle.tsx            # Alternative theme toggle
│   └── 📁 ui/                      # Shadcn/ui components
│       ├── accordion.tsx
│       ├── alert-dialog.tsx
│       ├── alert.tsx
│       ├── aspect-ratio.tsx
│       ├── avatar.tsx
│       ├── badge.tsx
│       ├── breadcrumb.tsx
│       ├── button.tsx
│       ├── calendar.tsx
│       ├── card.tsx
│       ├── carousel.tsx
│       ├── chart.tsx
│       ├── checkbox.tsx
│       ├── collapsible.tsx
│       ├── command.tsx
│       ├── context-menu.tsx
│       ├── dialog.tsx
│       ├── drawer.tsx
│       ├── dropdown-menu.tsx
│       ├── form.tsx
│       ├── hover-card.tsx
│       ├── input-otp.tsx
│       ├── input.tsx
│       ├── label.tsx
│       ├── menubar.tsx
│       ├── navigation-menu.tsx
│       ├── popover.tsx
│       ├── progress.tsx
│       ├── radio-group.tsx
│       ├── resizable.tsx
│       ├── scroll-area.tsx
│       ├── select.tsx
│       ├── separator.tsx
│       ├── sheet.tsx
│       ├── sidebar.tsx
│       ├── skeleton.tsx
│       ├── slider.tsx
│       ├── sonner.tsx
│       ├── switch.tsx
│       ├── table.tsx
│       ├── tabs.tsx
│       ├── textarea.tsx
│       ├── toast.tsx
│       ├── toaster.tsx
│       ├── toggle-group.tsx
│       ├── toggle.tsx
│       ├── tooltip.tsx
│       ├── use-mobile.tsx
│       └── use-toast.ts
├── 📁 fake-and-real-news-dataset/   # Training datasets
│   ├── Fake.csv
│   └── True.csv
├── 📁 hooks/                       # Custom React hooks
│   ├── use-mobile.tsx
│   ├── use-toast.ts
│   └── useFactCheck.ts             # Main fact-checking hook
├── 📁 liar_dataset/               # LIAR dataset
│   ├── README
│   ├── test.tsv
│   ├── train.tsv
│   └── valid.tsv
├── 📁 lib/                        # Utility libraries
│   ├── formatReasoning.tsx        # Reasoning formatting utilities
│   ├── openrouter.ts              # OpenRouter API client
│   └── utils.ts                   # General utilities
├── 📁 public/                     # Static assets
│   ├── index.html
│   ├── placeholder-logo.png
│   ├── placeholder-logo.svg
│   ├── placeholder-user.jpg
│   ├── placeholder.jpg
│   ├── placeholder.svg
│   ├── script.js
│   ├── style.css
│   └── 📁 models/                 # Frontend ML models
│       ├── preprocessing_config.json
│       ├── vocabulary.json
│       └── 📁 tfjs_model/
│           ├── group1-shard1of3.bin
│           ├── group1-shard2of3.bin
│           ├── group1-shard3of3.bin
│           └── model.json
├── 📁 src/                        # Source files
│   ├── prediction.js
│   ├── preprocessing.js
│   └── utils.js
├── 📁 styles/                     # Additional styles
│   └── globals.css
├── 📁 tests/                      # Test files
│   ├── test_model.js
│   └── test_preprocessing.js
├── 📁 training/                   # Model training infrastructure
│   ├── 📁 data/
│   │   ├── 📁 processed/
│   │   │   ├── feature_summary.json
│   │   │   ├── fully_processed_dataset.csv
│   │   │   ├── real_dataset_processed.csv
│   │   │   ├── tfidf_vectorizer_full.pkl
│   │   │   └── 📁 models/
│   │   │       ├── best_model_gradient_boosting.pkl
│   │   │       ├── best_model_metadata.json
│   │   │       ├── cnn_model.h5
│   │   │       ├── keras_tokenizer.pkl
│   │   │       ├── lstm_model.h5
│   │   │       └── 📁 version 2/
│   │   ├── 📁 raw/
│   │   └── 📁 New folder/
│   ├── 📁 notebooks/
│   │   ├── data_exploration.ipynb
│   │   ├── model_training.ipynb
│   │   └── preprocessing.ipynb
│   └── 📁 scripts/
│       ├── load_real_datasets.py
│       └── quick_data_check.py
├── 📄 .env.example                # Environment variables template
├── 📄 .env.local                  # Local environment variables
├── 📄 .gitignore                  # Git ignore rules
├── 📄 CHANGELOG.md                # Version history
├── 📄 components.json             # Shadcn/ui configuration
├── 📄 DEVELOPMENT_STATUS.md       # Development progress
├── 📄 INSTALL_GUIDE.md            # Installation instructions
├── 📄 INSTALL_GUIDE.md            # Installation instructions
├── 📄 MODEL_SELECTION_FEATURE.md  # Model selection documentation
├── 📄 README.md                   # This file
├── 📄 best_model_metadata.json    # Model metadata
├── 📄 netlify.toml                # Netlify deployment config
├── 📄 next.config.mjs             # Next.js configuration
├── 📄 package.json                # Frontend dependencies
├── 📄 plan.md                     # Project planning
├── 📄 pnpm-lock.yaml              # Package lock file
├── 📄 postcss.config.mjs          # PostCSS configuration
├── 📄 presentation.md             # Project presentation
├── 📄 requirements.txt            # Python dependencies
├── 📄 start-backend.bat           # Windows batch file to start backend
├── 📄 tailwind.config.ts          # Tailwind CSS configuration
├── 📄 test-model.js               # Model testing script
└── 📄 tsconfig.json               # TypeScript configuration
```

## ✨ Features

- **Multiple Model Support**: Choose between different AI models for analysis:
  - Gradient Boosting (traditional ML)
  - LSTM (deep learning)
  - CNN (deep learning)
- **Real-time Analysis**: Get instant feedback on the authenticity of text content
- **AI Cross-Verification**: Uses OpenRouter Mistral AI to provide additional verification
- **Detailed Reasoning**: View the AI's reasoning process for more transparent results
- **Modern UI**: Clean, responsive interface with glass-morphism design elements
- **Privacy-First**: Analysis happens directly in your browser

## 🚀 Getting Started

### Prerequisites

- Node.js (v18 or higher)
- Python (v3.8 or higher)
- pip (for Python package management)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/factradar.git
   cd factradar
   ```

2. Install frontend dependencies:
   ```bash
   npm install --legacy-peer-deps
   ```

3. Install backend dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Copy `.env.example` to `.env.local` and fill in your OpenRouter API key:
   ```bash
   cp .env.example .env.local
   # Edit .env.local with your actual API key
   ```

### Running the Application

1. **Start the backend server** (Terminal 1):
   ```bash
   cd backend
   python app.py
   ```

2. **Start the frontend** (Terminal 2):
   ```bash
   npm run dev
   ```

3. **Open your browser** and navigate to `http://localhost:3000`

## 🧠 How It Works

FactRadar uses a combination of traditional machine learning and deep learning techniques to analyze text content:

1. **Text Preprocessing**: Cleans and normalizes the input text
2. **Feature Extraction**: Extracts relevant features from the text
3. **Model Analysis**: Processes the features through the selected AI model
4. **AI Cross-Verification**: Sends the text to OpenRouter Mistral for additional verification
5. **Result Presentation**: Displays the analysis results with confidence scores

## 🤖 Models

### Gradient Boosting
Traditional ML model with high accuracy on structured features. Uses TF-IDF vectorization and engineered features.

**Performance Metrics:**
- Test Accuracy: 99.67%
- Test Precision: 100%
- Test Recall: 99.33%
- Test F1-Score: 99.67%

### LSTM (Long Short-Term Memory)
Deep learning model good at capturing sequential patterns in text data.

### CNN (Convolutional Neural Network)
Deep learning model effective at capturing local patterns and features in text.

## 🛠️ Tech Stack

### Frontend
- **Framework**: Next.js 15 with React 19
- **Styling**: Tailwind CSS + Shadcn/ui
- **Language**: TypeScript
- **State Management**: React Hooks
- **HTTP Client**: Fetch API

### Backend
- **Framework**: Flask (Python)
- **ML Libraries**: scikit-learn, TensorFlow, XGBoost
- **NLP**: NLTK
- **API**: RESTful endpoints

### Machine Learning
- **Traditional ML**: Gradient Boosting, XGBoost
- **Deep Learning**: LSTM, CNN
- **Feature Engineering**: TF-IDF, sentiment analysis, linguistic features

## 📊 Dataset Information

The models were trained on:
- **Total Samples**: 3,998 (1,999 real + 1,999 fake)
- **Training Set**: 2,798 samples
- **Validation Set**: 600 samples
- **Test Set**: 600 samples
- **Features**: 10,012 total features (10,000 TF-IDF + 12 engineered)

## 🔧 Development

### Project Scripts

**Frontend:**
```bash
npm run dev          # Start development server
npm run build        # Build for production
npm run start        # Start production server
npm run lint         # Run ESLint
```

**Backend:**
```bash
python app.py        # Start Flask server
python test.py       # Run backend tests
```

### Environment Variables

Create a `.env.local` file with:

```env
NEXT_PUBLIC_OPENROUTER_API_KEY=your_openrouter_api_key_here
BACKEND_URL=http://localhost:5000
NEXT_PUBLIC_FRONTEND_URL=http://localhost:3000
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Guidelines
1. Follow the existing code style
2. Add tests for new features
3. Update documentation as needed
4. Ensure all tests pass before submitting

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenRouter for providing AI verification capabilities
- Shadcn/ui for the beautiful UI components
- The open-source community for the amazing tools and libraries

## 📞 Support

For support, please open an issue on GitHub or contact the development team.