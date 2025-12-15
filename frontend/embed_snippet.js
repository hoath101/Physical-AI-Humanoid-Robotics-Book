/**
 * Physical AI & Humanoid Robotics Book RAG Chatbot Widget
 *
 * A minimal, framework-agnostic chatbot widget that can be embedded in Docusaurus sites.
 * Features:
 * - Text-based questions
 * - Selected-text-only mode
 * - Integration with FastAPI backend
 * - Responsive design
 */

(function() {
    'use strict';

    // Configuration - can be overridden by passing options to init function
    const CONFIG = {
        apiBaseUrl: window.CHATBOT_API_BASE_URL || 'http://localhost:8000',
        widgetTitle: 'Physical AI & Robotics Assistant',
        placeholderText: 'Ask about Physical AI, ROS 2, Humanoid Robotics...',
        selectedTextPlaceholder: 'Ask about the selected text...',
        botName: 'Robotics Assistant',
        userAvatar: '👤',
        botAvatar: '🤖',
        themeColor: '#4f46e5', // Indigo color
        maxHistoryItems: 50
    };

    // State management
    let state = {
        isOpen: false,
        selectedText: null,
        conversationHistory: [],
        isWaitingForResponse: false
    };

    // DOM elements cache
    let elements = {};

    /**
     * Initialize the chatbot widget
     * @param {Object} options - Configuration options
     */
    function initChatbot(options = {}) {
        // Merge options with default config
        Object.assign(CONFIG, options);

        // Create widget HTML
        createWidget();

        // Add event listeners
        addEventListeners();

        // Add global selection handler
        addGlobalSelectionHandler();

        console.log('Physical AI & Humanoid Robotics Chatbot initialized');
    }

    /**
     * Create the widget HTML structure
     */
    function createWidget() {
        // Create main container
        const widgetContainer = document.createElement('div');
        widgetContainer.id = 'chatbot-widget-container';
        widgetContainer.innerHTML = `
            <div id="chatbot-toggle-btn" style="
                position: fixed;
                bottom: 20px;
                right: 20px;
                width: 60px;
                height: 60px;
                border-radius: 50%;
                background-color: ${CONFIG.themeColor};
                color: white;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                z-index: 10000;
                font-size: 24px;
                transition: all 0.3s ease;
            ">
                🤖
            </div>
            <div id="chatbot-widget" style="
                position: fixed;
                bottom: 90px;
                right: 20px;
                width: 380px;
                max-height: 600px;
                background: white;
                border-radius: 12px;
                box-shadow: 0 10px 25px rgba(0,0,0,0.2);
                display: none;
                flex-direction: column;
                z-index: 10000;
                overflow: hidden;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            ">
                <div id="chatbot-header" style="
                    background-color: ${CONFIG.themeColor};
                    color: white;
                    padding: 16px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                ">
                    <div>
                        <div style="font-weight: 600; font-size: 16px;">${CONFIG.widgetTitle}</div>
                        <div style="font-size: 12px; opacity: 0.9;">Ask about Physical AI & Humanoid Robotics</div>
                    </div>
                    <button id="chatbot-close-btn" style="
                        background: none;
                        border: none;
                        color: white;
                        font-size: 20px;
                        cursor: pointer;
                        padding: 4px;
                        border-radius: 4px;
                    ">✕</button>
                </div>
                <div id="chatbot-messages" style="
                    flex: 1;
                    padding: 16px;
                    overflow-y: auto;
                    max-height: 400px;
                    background-color: #fafafa;
                "></div>
                <div id="chatbot-input-area" style="
                    padding: 12px;
                    background: white;
                    border-top: 1px solid #eee;
                ">
                    <div id="chatbot-selected-text-indicator" style="
                        display: none;
                        background-color: #dbeafe;
                        border: 1px solid #93c5fd;
                        border-radius: 6px;
                        padding: 8px;
                        margin-bottom: 8px;
                        font-size: 12px;
                        color: #1e40af;
                    ">
                        <strong>Selected text mode:</strong> <span id="selected-text-preview"></span>
                    </div>
                    <div style="display: flex; gap: 8px;">
                        <input
                            type="text"
                            id="chatbot-input"
                            placeholder="${CONFIG.placeholderText}"
                            style="
                                flex: 1;
                                padding: 10px 12px;
                                border: 1px solid #ddd;
                                border-radius: 6px;
                                font-size: 14px;
                            "
                        />
                        <button id="chatbot-send-btn" style="
                            background-color: ${CONFIG.themeColor};
                            color: white;
                            border: none;
                            border-radius: 6px;
                            padding: 10px 16px;
                            cursor: pointer;
                            font-weight: 500;
                        ">Send</button>
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(widgetContainer);

        // Cache elements for easier access
        elements = {
            container: document.getElementById('chatbot-widget-container'),
            toggleBtn: document.getElementById('chatbot-toggle-btn'),
            widget: document.getElementById('chatbot-widget'),
            closeBtn: document.getElementById('chatbot-close-btn'),
            messages: document.getElementById('chatbot-messages'),
            input: document.getElementById('chatbot-input'),
            sendBtn: document.getElementById('chatbot-send-btn'),
            selectedTextIndicator: document.getElementById('chatbot-selected-text-indicator'),
            selectedTextPreview: document.getElementById('selected-text-preview')
        };
    }

    /**
     * Add event listeners to the widget elements
     */
    function addEventListeners() {
        // Toggle widget visibility
        elements.toggleBtn.addEventListener('click', toggleWidget);
        elements.closeBtn.addEventListener('click', hideWidget);

        // Send message on button click
        elements.sendBtn.addEventListener('click', sendMessage);

        // Send message on Enter key (but allow Shift+Enter for new lines)
        elements.input.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        // Update placeholder based on selected text
        elements.input.addEventListener('focus', updateInputPlaceholder);
        elements.input.addEventListener('blur', updateInputPlaceholder);
    }

    /**
     * Add global selection handler to detect text selection
     */
    function addGlobalSelectionHandler() {
        document.addEventListener('mouseup', function() {
            const selectedText = getSelectedText();
            if (selectedText) {
                state.selectedText = selectedText.trim();
                updateSelectedTextIndicator();
            } else {
                state.selectedText = null;
                hideSelectedTextIndicator();
            }
        });
    }

    /**
     * Get currently selected text
     */
    function getSelectedText() {
        const selection = window.getSelection();
        if (selection.rangeCount > 0) {
            const range = selection.getRangeAt(0);
            const selectedText = range.toString().trim();
            return selectedText.length > 0 ? selectedText : null;
        }
        return null;
    }

    /**
     * Toggle widget visibility
     */
    function toggleWidget() {
        if (state.isOpen) {
            hideWidget();
        } else {
            showWidget();
        }
    }

    /**
     * Show the widget
     */
    function showWidget() {
        elements.widget.style.display = 'flex';
        state.isOpen = true;
        elements.input.focus();
        renderMessages();
    }

    /**
     * Hide the widget
     */
    function hideWidget() {
        elements.widget.style.display = 'none';
        state.isOpen = false;
    }

    /**
     * Update input placeholder based on selected text
     */
    function updateInputPlaceholder() {
        if (state.selectedText) {
            elements.input.placeholder = CONFIG.selectedTextPlaceholder;
        } else {
            elements.input.placeholder = CONFIG.placeholderText;
        }
    }

    /**
     * Show selected text indicator
     */
    function updateSelectedTextIndicator() {
        if (state.selectedText) {
            elements.selectedTextIndicator.style.display = 'block';
            elements.selectedTextPreview.textContent = state.selectedText.length > 80
                ? state.selectedText.substring(0, 80) + '...'
                : state.selectedText;
        }
    }

    /**
     * Hide selected text indicator
     */
    function hideSelectedTextIndicator() {
        elements.selectedTextIndicator.style.display = 'none';
    }

    /**
     * Send a message to the backend
     */
    async function sendMessage() {
        const message = elements.input.value.trim();
        if (!message || state.isWaitingForResponse) {
            return;
        }

        // Add user message to UI
        addMessageToUI('user', message);

        // Clear input
        elements.input.value = '';

        // Show typing indicator
        const typingIndicator = addMessageToUI('bot', '...');
        state.isWaitingForResponse = true;

        try {
            // Prepare the request payload
            const requestBody = {
                question: message,
                selected_text: state.selectedText || null
            };

            // Call the backend API
            const response = await fetch(`${CONFIG.apiBaseUrl}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestBody)
            });

            if (!response.ok) {
                throw new Error(`API request failed with status ${response.status}`);
            }

            const data = await response.json();

            // Remove typing indicator
            removeMessageFromUI(typingIndicator);

            // Add bot response to UI
            addMessageToUI('bot', data.answer);

            // If we were in selected-text mode, clear it after the response
            if (state.selectedText) {
                state.selectedText = null;
                hideSelectedTextIndicator();
                updateInputPlaceholder();
            }

        } catch (error) {
            console.error('Error sending message:', error);

            // Remove typing indicator
            removeMessageFromUI(typingIndicator);

            // Show error message
            addMessageToUI('bot', `Sorry, I encountered an error: ${error.message}`);
        } finally {
            state.isWaitingForResponse = false;
        }
    }

    /**
     * Add a message to the UI
     */
    function addMessageToUI(sender, text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chatbot-message chatbot-message-${sender}`;
        messageDiv.style.cssText = `
            display: flex;
            margin-bottom: 12px;
            align-items: flex-start;
        `;

        const avatarDiv = document.createElement('div');
        avatarDiv.textContent = sender === 'user' ? CONFIG.userAvatar : CONFIG.botAvatar;
        avatarDiv.style.cssText = `
            width: 30px;
            height: 30px;
            border-radius: 50%;
            background-color: ${sender === 'user' ? '#e2e8f0' : '#dbeafe'};
            color: ${sender === 'user' ? '#4a5568' : '#1e40af'};
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            margin-right: 8px;
            font-size: 14px;
        `;

        const contentDiv = document.createElement('div');
        contentDiv.style.cssText = `
            background-color: ${sender === 'user' ? '#e2e8f0' : '#f0f9ff'};
            padding: 10px 12px;
            border-radius: 18px;
            max-width: 80%;
            line-height: 1.4;
            font-size: 14px;
        `;

        // Process text for better formatting (convert newlines to <br>)
        const formattedText = text.replace(/\n/g, '<br>');
        contentDiv.innerHTML = formattedText;

        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(contentDiv);

        elements.messages.appendChild(messageDiv);

        // Scroll to bottom
        elements.messages.scrollTop = elements.messages.scrollHeight;

        // Store in history
        const messageObj = { sender, text, timestamp: Date.now() };
        state.conversationHistory.push(messageObj);

        // Limit history size
        if (state.conversationHistory.length > CONFIG.maxHistoryItems) {
            state.conversationHistory.shift();
        }

        return messageDiv;
    }

    /**
     * Remove a message from the UI
     */
    function removeMessageFromUI(messageElement) {
        if (messageElement && messageElement.parentNode) {
            messageElement.parentNode.removeChild(messageElement);
        }
    }

    /**
     * Render all messages in the conversation history
     */
    function renderMessages() {
        elements.messages.innerHTML = '';
        state.conversationHistory.forEach(msg => {
            addMessageToUI(msg.sender, msg.text);
        });

        // Scroll to bottom
        elements.messages.scrollTop = elements.messages.scrollHeight;
    }

    /**
     * Public API for the chatbot
     */
    window.PhysicalAIRoboticsChatbot = {
        init: initChatbot,
        show: showWidget,
        hide: hideWidget,
        toggle: toggleWidget,
        sendMessage: sendMessage,
        addMessage: addMessageToUI,
        clearHistory: function() {
            state.conversationHistory = [];
            elements.messages.innerHTML = '';
        }
    };

    // Auto-initialize if config is available
    if (window.CHATBOT_CONFIG) {
        window.PhysicalAIRoboticsChatbot.init(window.CHATBOT_CONFIG);
    }

})();