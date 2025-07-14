// FactRadar Main Application
class FactRadar {
  constructor() {
    this.currentSection = "home"
    this.isAnalyzing = false
    this.sampleTexts = {
      climate:
        "Scientists from NASA have released new data showing continued warming trends across global temperature measurements. The comprehensive study, published in the Journal of Climate Science, analyzed temperature records from over 6,000 weather stations worldwide spanning the past 50 years. The research indicates that global average temperatures have risen by 1.2 degrees Celsius since pre-industrial times, with the most significant increases observed in Arctic regions. The study's lead author, Dr. Sarah Chen, emphasized that these findings align with previous climate models and underscore the urgent need for continued climate action.",
      tech: "Researchers at MIT have developed a new battery technology that could revolutionize electric vehicle charging times. The breakthrough involves a novel lithium-metal battery design that can charge to 80% capacity in just 10 minutes while maintaining safety and longevity. The technology uses a unique polymer coating that prevents dangerous dendrite formation, a major obstacle in previous lithium-metal battery designs. Initial tests show the batteries can maintain 90% of their capacity after 10,000 charge cycles. The research team, led by Professor Angela Zhang, expects commercial applications to be available within 3-5 years.",
      miracle:
        "Local doctor discovers simple trick that cures all diseases using this one weird ingredient that pharmaceutical companies don't want you to know about! Dr. Johnson from downtown clinic has been secretly treating patients with this amazing natural remedy that Big Pharma has been hiding from the public for decades. Patients report miraculous recoveries from cancer, diabetes, heart disease, and even aging itself! The secret ingredient, found in every kitchen, has been suppressing illness for centuries but medical establishments refuse to acknowledge its power. Click here to learn the shocking truth that doctors hate!",
      satirical:
        "Local resident shocked to learn that other people also have opinions, plans to write strongly worded social media posts about it. Area man Derek Thompson, 34, was reportedly 'absolutely flabbergasted' to discover Tuesday that other human beings possess their own thoughts and viewpoints that may differ from his own. 'I just assumed everyone thought exactly like me,' said Thompson, frantically typing his 47th Facebook post of the day. 'But apparently some people have different experiences and perspectives. This is outrageous!' Thompson has since started a petition demanding that all humans think identically to avoid future confusion.",
      mixed:
        "Studies show that drinking green tea can boost metabolism by 400%. My friend tried it and lost 20 pounds in a week! Scientists are amazed by these results. While green tea does contain compounds like EGCG that may have metabolic benefits, the extreme claims about 400% metabolism boost are not supported by peer-reviewed research. Some studies suggest modest improvements in fat oxidation, but realistic expectations should be around 4-5% increase in metabolic rate, not 400%. The anecdotal evidence about rapid weight loss is likely due to other factors and should be verified through controlled studies.",
    }

    this.init()
  }

  init() {
    this.bindEvents()
    this.updateCharCounter()
    this.initializeNavigation()
    this.initializeTouchSupport()

    // Initialize page-specific features when navigating
    this.initializePageFeatures()
  }

  bindEvents() {
    // Input events
    const newsInput = document.getElementById("newsInput")
    const analyzeBtn = document.getElementById("analyzeBtn")
    const clearTextBtn = document.getElementById("clearTextBtn")
    const trySampleBtn = document.getElementById("trySampleBtn")

    newsInput.addEventListener("input", () => {
      this.updateCharCounter()
      this.toggleAnalyzeButton()
    })

    newsInput.addEventListener("paste", () => {
      setTimeout(() => {
        this.updateCharCounter()
        this.toggleAnalyzeButton()
        this.showPasteAnimation()
      }, 10)
    })

    analyzeBtn.addEventListener("click", () => this.analyzeText())
    clearTextBtn.addEventListener("click", () => this.clearText())
    trySampleBtn.addEventListener("click", () => this.showRandomSample())

    // Modal events
    const closeModal = document.getElementById("closeModal")
    const modalOverlay = document.getElementById("resultsModal")
    const tryAnotherBtn = document.getElementById("tryAnotherBtn")

    closeModal.addEventListener("click", () => this.closeModal())
    modalOverlay.addEventListener("click", (e) => {
      if (e.target === modalOverlay) this.closeModal()
    })
    tryAnotherBtn.addEventListener("click", () => this.closeModal())

    // Navigation events
    document.querySelectorAll(".nav-link").forEach((link) => {
      link.addEventListener("click", (e) => {
        e.preventDefault()
        const section = link.getAttribute("href").substring(1)
        this.navigateToSection(section)
      })
    })

    // Sample tabs
    document.querySelectorAll(".tab-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const tab = btn.getAttribute("data-tab")
        this.switchSampleTab(tab)
      })
    })

    // Sample buttons
    document.querySelectorAll(".btn-sample").forEach((btn) => {
      btn.addEventListener("click", () => {
        const sample = btn.getAttribute("data-sample")
        this.loadSample(sample)
      })
    })

    // Keyboard shortcuts
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && !document.getElementById("resultsModal").classList.contains("hidden")) {
        this.closeModal()
      }
      if (e.ctrlKey && e.key === "Enter") {
        if (!this.isAnalyzing && document.getElementById("newsInput").value.trim()) {
          this.analyzeText()
        }
      }
    })

    // Add after existing event bindings

    // Category tab events for samples page
    document.querySelectorAll(".category-tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        const category = tab.getAttribute("data-category")
        this.switchSampleCategory(category)
      })
    })

    // FAQ toggle events
    document.querySelectorAll(".faq-question").forEach((question) => {
      question.addEventListener("click", () => {
        const faqItem = question.parentElement
        faqItem.classList.toggle("active")
      })
    })

    // Contact form submission
    const contactForm = document.getElementById("contactForm")
    if (contactForm) {
      contactForm.addEventListener("submit", (e) => {
        e.preventDefault()
        this.handleContactForm(e)
      })
    }

    // Process step animations on scroll
    this.initializeProcessAnimations()

    // Performance chart animations
    this.initializePerformanceCharts()
  }

  updateCharCounter() {
    const newsInput = document.getElementById("newsInput")
    const charCounter = document.querySelector(".char-counter")
    const currentLength = newsInput.value.length
    const maxLength = 2000

    charCounter.textContent = `${currentLength} / ${maxLength}`

    if (currentLength > maxLength * 0.9) {
      charCounter.style.color = "#ef4444"
    } else if (currentLength > maxLength * 0.7) {
      charCounter.style.color = "#f59e0b"
    } else {
      charCounter.style.color = "rgba(255, 255, 255, 0.6)"
    }
  }

  toggleAnalyzeButton() {
    const newsInput = document.getElementById("newsInput")
    const analyzeBtn = document.getElementById("analyzeBtn")
    const hasText = newsInput.value.trim().length > 0

    analyzeBtn.disabled = !hasText || this.isAnalyzing
  }

  showPasteAnimation() {
    const inputContainer = document.querySelector(".input-container")
    inputContainer.style.transform = "scale(1.02)"
    inputContainer.style.borderColor = "rgba(102, 126, 234, 0.5)"

    setTimeout(() => {
      inputContainer.style.transform = "scale(1)"
      inputContainer.style.borderColor = ""
    }, 200)
  }

  clearText() {
    const newsInput = document.getElementById("newsInput")
    newsInput.value = ""
    this.updateCharCounter()
    this.toggleAnalyzeButton()
    newsInput.focus()
  }

  showRandomSample() {
    const samples = Object.keys(this.sampleTexts)
    const randomSample = samples[Math.floor(Math.random() * samples.length)]
    this.loadSample(randomSample)
  }

  loadSample(sampleKey) {
    const newsInput = document.getElementById("newsInput")
    const sampleText = this.sampleTexts[sampleKey]

    if (sampleText) {
      newsInput.value = sampleText
      this.updateCharCounter()
      this.toggleAnalyzeButton()

      // Scroll to input if not visible
      if (this.currentSection !== "home") {
        this.navigateToSection("home")
      }

      setTimeout(() => {
        newsInput.scrollIntoView({ behavior: "smooth", block: "center" })
      }, 300)
    }
  }

  async analyzeText() {
    if (this.isAnalyzing) return

    const newsInput = document.getElementById("newsInput")
    const text = newsInput.value.trim()

    if (!text) return

    this.isAnalyzing = true
    this.showModal()
    this.showLoadingState()

    try {
      // Simulate API call with realistic delay
      await this.simulateAnalysis(text)

      // Generate mock results
      const results = this.generateMockResults(text)

      // Show results
      this.showResults(results)
    } catch (error) {
      console.error("Analysis error:", error)
      this.showError("Analysis failed. Please try again.")
    } finally {
      this.isAnalyzing = false
    }
  }

  async simulateAnalysis(text) {
    const steps = [
      "Processing text patterns...",
      "Analyzing language structure...",
      "Checking credibility indicators...",
      "Calculating confidence score...",
      "Generating report...",
    ]

    for (let i = 0; i < steps.length; i++) {
      document.querySelector(".loading-text").textContent = steps[i]
      await new Promise((resolve) => setTimeout(resolve, 800 + Math.random() * 400))
    }
  }

  generateMockResults(text) {
    // Simple heuristics for demo purposes
    const suspiciousWords = ["miracle", "secret", "doctors hate", "one weird trick", "shocking truth", "big pharma"]
    const credibleSources = ["nasa", "mit", "university", "journal", "research", "study", "professor", "dr."]
    const satiricalWords = ["area man", "local resident", "reportedly", "sources say", "shocking"]

    const lowerText = text.toLowerCase()

    let suspiciousCount = 0
    let credibleCount = 0
    let satiricalCount = 0

    suspiciousWords.forEach((word) => {
      if (lowerText.includes(word)) suspiciousCount++
    })

    credibleSources.forEach((word) => {
      if (lowerText.includes(word)) credibleCount++
    })

    satiricalWords.forEach((word) => {
      if (lowerText.includes(word)) satiricalCount++
    })

    let verdict, confidence, explanation, recommendation

    if (satiricalCount >= 2) {
      verdict = "satirical"
      confidence = 85 + Math.random() * 10
      explanation = [
        "Contains typical satirical news language patterns",
        "Uses exaggerated scenarios common in humor writing",
        "Lacks serious news reporting structure",
      ]
      recommendation = "This appears to be satirical content for entertainment purposes"
    } else if (suspiciousCount >= 2) {
      verdict = "fake"
      confidence = 75 + Math.random() * 20
      explanation = [
        "Contains multiple suspicious phrases commonly found in misinformation",
        "Uses emotional manipulation tactics",
        "Lacks credible source citations",
        "Makes extraordinary claims without evidence",
      ]
      recommendation = "High likelihood of misinformation. Verify with trusted sources"
    } else if (credibleCount >= 2) {
      verdict = "real"
      confidence = 80 + Math.random() * 15
      explanation = [
        "References credible institutions and sources",
        "Uses professional journalism language",
        "Contains specific, verifiable details",
        "Follows standard news reporting structure",
      ]
      recommendation = "Appears to be from a credible source, but always cross-check important information"
    } else {
      verdict = "uncertain"
      confidence = 40 + Math.random() * 30
      explanation = [
        "Mixed indicators present in the text",
        "Insufficient information to make confident determination",
        "May require additional context or verification",
      ]
      recommendation = "Unable to determine reliability. Recommend checking multiple sources"
    }

    // Add suspicious phrases detection
    const suspiciousPhrases = []
    const phrases = [
      "one weird trick",
      "doctors hate",
      "big pharma",
      "secret ingredient",
      "miracle cure",
      "shocking truth",
      "they don't want you to know",
    ]

    phrases.forEach((phrase) => {
      const regex = new RegExp(phrase, "gi")
      const matches = text.match(regex)
      if (matches) {
        suspiciousPhrases.push(...matches)
      }
    })

    return {
      verdict,
      confidence: Math.round(confidence),
      credibility: Math.round(confidence * 0.9),
      sentiment: this.analyzeSentiment(text),
      readability: this.analyzeReadability(text),
      sourceReliability: this.analyzeSourceReliability(text),
      suspiciousPhrases,
      explanation,
      recommendation,
    }
  }

  analyzeSentiment(text) {
    const positiveWords = ["good", "great", "excellent", "positive", "success", "breakthrough", "amazing"]
    const negativeWords = ["bad", "terrible", "awful", "negative", "failure", "disaster", "shocking"]

    const lowerText = text.toLowerCase()
    let positiveCount = 0
    let negativeCount = 0

    positiveWords.forEach((word) => {
      if (lowerText.includes(word)) positiveCount++
    })

    negativeWords.forEach((word) => {
      if (lowerText.includes(word)) negativeCount++
    })

    if (positiveCount > negativeCount) return "positive"
    if (negativeCount > positiveCount) return "negative"
    return "neutral"
  }

  analyzeReadability(text) {
    const words = text.split(/\s+/).length
    const sentences = text.split(/[.!?]+/).length
    const avgWordsPerSentence = words / sentences

    if (avgWordsPerSentence < 15) return "easy"
    if (avgWordsPerSentence < 25) return "medium"
    return "hard"
  }

  showModal() {
    const modal = document.getElementById("resultsModal")
    modal.classList.remove("hidden")
    document.body.style.overflow = "hidden"
  }

  closeModal() {
    const modal = document.getElementById("resultsModal")
    modal.classList.add("hidden")
    document.body.style.overflow = ""

    // Reset modal state
    document.getElementById("loadingState").classList.remove("hidden")
    document.getElementById("resultsDisplay").classList.add("hidden")
  }

  showLoadingState() {
    document.getElementById("loadingState").classList.remove("hidden")
    document.getElementById("resultsDisplay").classList.add("hidden")
  }

  showResults(results) {
    // Hide loading, show results
    document.getElementById("loadingState").classList.add("hidden")
    document.getElementById("resultsDisplay").classList.remove("hidden")

    // Update verdict
    const statusCircle = document.querySelector(".status-circle")
    const verdictText = document.getElementById("verdictText")

    statusCircle.className = `status-circle ${results.verdict}`

    switch (results.verdict) {
      case "real":
        verdictText.textContent = "LIKELY REAL"
        statusCircle.innerHTML = "✓"
        break
      case "fake":
        verdictText.textContent = "LIKELY FAKE"
        statusCircle.innerHTML = "✗"
        break
      case "satirical":
        verdictText.textContent = "SATIRICAL"
        statusCircle.innerHTML = "😄"
        break
      default:
        verdictText.textContent = "UNCERTAIN"
        statusCircle.innerHTML = "?"
    }

    // Animate confidence score
    this.animateConfidence(results.confidence)

    // Update indicators
    this.updateIndicators(results)

    // Update explanation
    this.updateExplanation(results)

    // Add source reliability indicator
    const sourceReliability = document.createElement("div")
    sourceReliability.className = "indicator"
    sourceReliability.innerHTML = `
      <span class="indicator-label">Source</span>
      <span class="source-badge ${results.sourceReliability}">${results.sourceReliability.charAt(0).toUpperCase() + results.sourceReliability.slice(1)}</span>
    `
    document.querySelector(".indicators-grid").appendChild(sourceReliability)

    // Add suspicious phrases if any
    if (results.suspiciousPhrases && results.suspiciousPhrases.length > 0) {
      const phrasesSection = document.createElement("div")
      phrasesSection.className = "suspicious-phrases"
      phrasesSection.innerHTML = `
        <h4>Suspicious Phrases Detected</h4>
        <div class="phrases-list">
          ${results.suspiciousPhrases.map((phrase) => `<span class="suspicious-phrase">"${phrase}"</span>`).join("")}
        </div>
      `
      document.querySelector(".explanation-card").appendChild(phrasesSection)
    }
  }

  animateConfidence(confidence) {
    const confidenceFill = document.getElementById("confidenceFill")
    const confidencePercentage = document.getElementById("confidencePercentage")

    // Reset
    confidenceFill.style.width = "0%"
    confidencePercentage.textContent = "0%"

    // Animate
    setTimeout(() => {
      confidenceFill.style.width = `${confidence}%`

      let current = 0
      const increment = confidence / 50
      const timer = setInterval(() => {
        current += increment
        if (current >= confidence) {
          current = confidence
          clearInterval(timer)
        }
        confidencePercentage.textContent = `${Math.round(current)}%`
      }, 20)
    }, 500)
  }

  updateIndicators(results) {
    // Credibility meter
    const credibilityMeter = document.getElementById("credibilityMeter")
    setTimeout(() => {
      credibilityMeter.style.width = `${results.credibility}%`
    }, 800)

    // Sentiment badge
    const sentimentBadge = document.getElementById("sentimentBadge")
    sentimentBadge.textContent = results.sentiment.charAt(0).toUpperCase() + results.sentiment.slice(1)
    sentimentBadge.className = `sentiment-badge ${results.sentiment}`

    // Readability badge
    const readabilityBadge = document.getElementById("readabilityBadge")
    readabilityBadge.textContent = results.readability.charAt(0).toUpperCase() + results.readability.slice(1)
    readabilityBadge.className = `readability-badge ${results.readability}`
  }

  updateExplanation(results) {
    const explanationList = document.getElementById("explanationList")
    const recommendation = document.getElementById("recommendation")

    // Clear existing content
    explanationList.innerHTML = ""

    // Add explanation points with staggered animation
    results.explanation.forEach((point, index) => {
      setTimeout(() => {
        const li = document.createElement("li")
        li.textContent = point
        li.style.opacity = "0"
        li.style.transform = "translateY(10px)"
        explanationList.appendChild(li)

        setTimeout(() => {
          li.style.transition = "all 0.3s ease"
          li.style.opacity = "1"
          li.style.transform = "translateY(0)"
        }, 50)
      }, index * 200)
    })

    // Update recommendation
    setTimeout(
      () => {
        recommendation.querySelector("span").textContent = results.recommendation
      },
      results.explanation.length * 200 + 300,
    )
  }

  showError(message) {
    document.getElementById("loadingState").classList.add("hidden")
    document.getElementById("resultsDisplay").innerHTML = `
            <div class="error-state">
                <div class="error-icon">
                    <i class="fas fa-exclamation-triangle"></i>
                </div>
                <h3>Analysis Error</h3>
                <p>${message}</p>
                <button class="btn-primary" onclick="factRadar.closeModal()">Try Again</button>
            </div>
        `
    document.getElementById("resultsDisplay").classList.remove("hidden")
  }

  initializeNavigation() {
    // Set initial active nav link
    this.updateActiveNavLink("home")
  }

  navigateToSection(section) {
    // Hide all sections
    document.querySelectorAll("main > section, #home").forEach((el) => {
      el.classList.add("hidden")
    })

    // Show target section
    const targetSection = document.getElementById(section)
    if (targetSection) {
      targetSection.classList.remove("hidden")
      this.currentSection = section
      this.updateActiveNavLink(section)

      // Smooth scroll to top
      window.scrollTo({ top: 0, behavior: "smooth" })
    }
  }

  updateActiveNavLink(activeSection) {
    document.querySelectorAll(".nav-link").forEach((link) => {
      link.classList.remove("active")
      if (link.getAttribute("href") === `#${activeSection}`) {
        link.classList.add("active")
      }
    })
  }

  switchSampleTab(tabName) {
    // Update tab buttons
    document.querySelectorAll(".tab-btn").forEach((btn) => {
      btn.classList.remove("active")
      if (btn.getAttribute("data-tab") === tabName) {
        btn.classList.add("active")
      }
    })

    // Update tab content
    document.querySelectorAll(".tab-content").forEach((content) => {
      content.classList.remove("active")
      if (content.id === `${tabName}-tab`) {
        content.classList.add("active")
      }
    })
  }

  switchSampleCategory(category) {
    // Update active tab
    document.querySelectorAll(".category-tab").forEach((tab) => {
      tab.classList.remove("active")
      if (tab.getAttribute("data-category") === category) {
        tab.classList.add("active")
      }
    })

    // Update active content
    document.querySelectorAll(".category-content").forEach((content) => {
      content.classList.remove("active")
      const targetContent = document.getElementById(`${category}-category`)
      if (targetContent) {
        targetContent.classList.add("active")
      }
    })
  }

  handleContactForm(e) {
    const formData = new FormData(e.target)
    const data = Object.fromEntries(formData)

    // Show loading state
    const submitBtn = e.target.querySelector('button[type="submit"]')
    const originalText = submitBtn.innerHTML
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Sending...'
    submitBtn.disabled = true

    // Simulate form submission
    setTimeout(() => {
      // Show success message
      this.showNotification("Message sent successfully! We'll get back to you soon.", "success")

      // Reset form
      e.target.reset()

      // Reset button
      submitBtn.innerHTML = originalText
      submitBtn.disabled = false
    }, 2000)
  }

  showNotification(message, type = "info") {
    // Create notification element
    const notification = document.createElement("div")
    notification.className = `notification notification-${type}`
    notification.innerHTML = `
      <div class="notification-content">
        <i class="fas fa-${type === "success" ? "check-circle" : "info-circle"}"></i>
        <span>${message}</span>
      </div>
      <button class="notification-close">
        <i class="fas fa-times"></i>
      </button>
    `

    // Add to page
    document.body.appendChild(notification)

    // Add close functionality
    notification.querySelector(".notification-close").addEventListener("click", () => {
      notification.remove()
    })

    // Auto remove after 5 seconds
    setTimeout(() => {
      if (notification.parentElement) {
        notification.remove()
      }
    }, 5000)

    // Animate in
    setTimeout(() => {
      notification.classList.add("show")
    }, 100)
  }

  initializeProcessAnimations() {
    const processSteps = document.querySelectorAll(".process-step")

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.style.animationPlayState = "running"
          }
        })
      },
      { threshold: 0.3 },
    )

    processSteps.forEach((step) => {
      step.style.animationPlayState = "paused"
      observer.observe(step)
    })
  }

  initializePerformanceCharts() {
    const charts = document.querySelectorAll(".chart-circle")

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const chart = entry.target
            const percentage = Number.parseFloat(chart.getAttribute("data-percentage"))
            const degrees = (percentage / 100) * 360

            chart.style.setProperty("--percentage", `${degrees}deg`)

            // Animate the percentage counter
            this.animateCounter(chart.querySelector(".chart-percentage"), 0, percentage, 2000)
          }
        })
      },
      { threshold: 0.5 },
    )

    charts.forEach((chart) => observer.observe(chart))
  }

  animateCounter(element, start, end, duration) {
    const startTime = performance.now()

    const updateCounter = (currentTime) => {
      const elapsed = currentTime - startTime
      const progress = Math.min(elapsed / duration, 1)

      const current = start + (end - start) * this.easeOutCubic(progress)
      element.textContent = `${current.toFixed(1)}%`

      if (progress < 1) {
        requestAnimationFrame(updateCounter)
      }
    }

    requestAnimationFrame(updateCounter)
  }

  easeOutCubic(t) {
    return 1 - Math.pow(1 - t, 3)
  }

  analyzeSourceReliability(text) {
    const trustedSources = ["nasa", "mit", "harvard", "stanford", "reuters", "ap news", "bbc"]
    const questionableSources = ["blog", "social media", "unknown", "forwarded message"]

    const lowerText = text.toLowerCase()

    for (const source of trustedSources) {
      if (lowerText.includes(source)) return "trusted"
    }

    for (const source of questionableSources) {
      if (lowerText.includes(source)) return "questionable"
    }

    return "unknown"
  }

  // Add touch/swipe support for mobile
  initializeTouchSupport() {
    let startY = 0
    let startX = 0

    const modal = document.getElementById("resultsModal")

    modal.addEventListener(
      "touchstart",
      (e) => {
        startY = e.touches[0].clientY
        startX = e.touches[0].clientX
      },
      { passive: true },
    )

    modal.addEventListener(
      "touchmove",
      (e) => {
        if (!startY || !startX) return

        const currentY = e.touches[0].clientY
        const currentX = e.touches[0].clientX

        const diffY = startY - currentY
        const diffX = startX - currentX

        // Swipe down to close
        if (Math.abs(diffY) > Math.abs(diffX) && diffY < -100) {
          this.closeModal()
        }
      },
      { passive: true },
    )

    modal.addEventListener(
      "touchend",
      () => {
        startY = 0
        startX = 0
      },
      { passive: true },
    )
  }

  // Add social sharing functionality
  shareResults(results) {
    const shareData = {
      title: "FactRadar Analysis Results",
      text: `I just analyzed some news with FactRadar AI. Result: ${results.verdict.toUpperCase()} with ${results.confidence}% confidence.`,
      url: window.location.href,
    }

    if (navigator.share) {
      navigator.share(shareData)
    } else {
      // Fallback: copy to clipboard
      navigator.clipboard.writeText(`${shareData.text} ${shareData.url}`).then(() => {
        this.showNotification("Results copied to clipboard!", "success")
      })
    }
  }

  initializePageFeatures() {
    // Initialize FAQ toggles
    document.querySelectorAll(".faq-question").forEach((question) => {
      question.addEventListener("click", () => {
        const faqItem = question.parentElement
        faqItem.classList.toggle("active")
      })
    })
  }
}

// Initialize the application
const factRadar = new FactRadar()

// Add some additional utility functions
document.addEventListener("DOMContentLoaded", () => {
  // Add smooth scrolling for anchor links
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", function (e) {
      e.preventDefault()
      const target = document.querySelector(this.getAttribute("href"))
      if (target) {
        target.scrollIntoView({
          behavior: "smooth",
          block: "start",
        })
      }
    })
  })

  // Add intersection observer for scroll animations
  const observerOptions = {
    threshold: 0.1,
    rootMargin: "0px 0px -50px 0px",
  }

  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.style.opacity = "1"
        entry.target.style.transform = "translateY(0)"
      }
    })
  }, observerOptions)

  // Observe feature cards and other animated elements
  document.querySelectorAll(".feature-card, .about-card, .sample-card").forEach((card) => {
    card.style.opacity = "0"
    card.style.transform = "translateY(20px)"
    card.style.transition = "all 0.6s ease"
    observer.observe(card)
  })

  // Add mobile menu functionality
  const mobileMenuToggle = document.querySelector(".mobile-menu-toggle")
  const navMenu = document.querySelector(".nav-menu")

  if (mobileMenuToggle && navMenu) {
    mobileMenuToggle.addEventListener("click", () => {
      navMenu.classList.toggle("mobile-active")
      const icon = mobileMenuToggle.querySelector("i")
      icon.classList.toggle("fa-bars")
      icon.classList.toggle("fa-times")
    })

    // Close mobile menu when clicking nav links
    document.querySelectorAll(".nav-link").forEach((link) => {
      link.addEventListener("click", () => {
        navMenu.classList.remove("mobile-active")
        const icon = mobileMenuToggle.querySelector("i")
        icon.classList.add("fa-bars")
        icon.classList.remove("fa-times")
      })
    })
  }

  // Add keyboard navigation support
  document.addEventListener("keydown", (e) => {
    // Tab navigation enhancement
    if (e.key === "Tab") {
      document.body.classList.add("keyboard-navigation")
    }
  })

  document.addEventListener("mousedown", () => {
    document.body.classList.remove("keyboard-navigation")
  })

  // Add focus styles for keyboard navigation
  const style = document.createElement("style")
  style.textContent = `
        .keyboard-navigation *:focus {
            outline: 2px solid #667eea !important;
            outline-offset: 2px !important;
        }
        
        @media (max-width: 768px) {
            .nav-menu.mobile-active {
                display: flex !important;
                position: absolute;
                top: 100%;
                left: 0;
                right: 0;
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(20px);
                flex-direction: column;
                padding: 1rem;
                border-top: 1px solid rgba(255, 255, 255, 0.1);
                animation: mobileMenuSlide 0.3s ease-out;
            }
            
            @keyframes mobileMenuSlide {
                from {
                    opacity: 0;
                    transform: translateY(-10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
        }
        
        .error-state {
            text-align: center;
            padding: 3rem 0;
        }
        
        .error-icon {
            font-size: 3rem;
            color: #ef4444;
            margin-bottom: 1rem;
        }
        
        .error-state h3 {
            color: #ffffff;
            font-size: 1.5rem;
            margin-bottom: 1rem;
        }
        
        .error-state p {
            color: rgba(255, 255, 255, 0.8);
            margin-bottom: 2rem;
        }
    `
  document.head.appendChild(style)
})

// Add performance monitoring
if ("performance" in window) {
  window.addEventListener("load", () => {
    setTimeout(() => {
      const perfData = performance.getEntriesByType("navigation")[0]
      console.log(`Page load time: ${perfData.loadEventEnd - perfData.loadEventStart}ms`)
    }, 0)
  })
}

// Add error handling for uncaught errors
window.addEventListener("error", (e) => {
  console.error("Application error:", e.error)
  // Could send to error reporting service in production
})

// Export for potential module use
if (typeof module !== "undefined" && module.exports) {
  module.exports = FactRadar
}
