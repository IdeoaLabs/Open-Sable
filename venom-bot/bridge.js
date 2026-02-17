/**
 * Venom Bot Bridge for OpenSable
 * 
 * Provides WhatsApp Web automation via Venom Bot library
 * Communicates with Python backend via HTTP API
 */

const venom = require('venom-bot');
const express = require('express');
const bodyParser = require('body-parser');

// Parse command line arguments
const args = process.argv.slice(2);
const sessionName = args[args.indexOf('--session') + 1] || 'opensable';
const port = parseInt(args[args.indexOf('--port') + 1]) || 3333;

let client = null;

// Initialize Express server for API
const app = express();
app.use(bodyParser.json());

/**
 * Send event to Python backend via console (JSON format)
 */
function sendEvent(type, data = {}) {
    console.log(JSON.stringify({ type, data }));
}

/**
 * Initialize Venom Bot client
 */
async function initVenom() {
    try {
        sendEvent('info', { message: 'Initializing Venom Bot...' });
        
        client = await venom.create(
            sessionName,
            (base64Qr, asciiQR) => {
                // QR Code received - display in terminal
                console.log('\n\n========== SCAN THIS QR CODE ==========\n');
                console.log(asciiQR);
                console.log('\n========================================\n');
                sendEvent('qr', { qr: asciiQR, base64: base64Qr });
            },
            (statusSession) => {
                sendEvent('status', { status: statusSession });
            },
            {
                headless: false,
                devtools: false,
                useChrome: true,
                debug: false,
                logQR: true,
                browserArgs: [
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-accelerated-2d-canvas',
                    '--no-first-run',
                    '--no-zygote',
                    '--disable-gpu'
                ],
                autoClose: 60000,
                disableSpins: true,
            }
        );
        
        sendEvent('authenticated', { session: sessionName });
        
        // Start listening for messages
        await startMessageListener();
        
        sendEvent('ready', { message: 'WhatsApp bot is ready!' });
        
    } catch (error) {
        sendEvent('error', { error: error.message, stack: error.stack });
        process.exit(1);
    }
}

/**
 * Listen for incoming WhatsApp messages
 */
async function startMessageListener() {
    client.onMessage(async (message) => {
        try {
            // Send message event to Python
            sendEvent('message', { data: message });
            
        } catch (error) {
            sendEvent('error', { error: error.message, context: 'message_handler' });
        }
    });
    
    // Listen for message acknowledgements
    client.onAck(async (ack) => {
        sendEvent('ack', { data: ack });
    });
    
    // Listen for state changes
    client.onStateChange((state) => {
        sendEvent('state_change', { state });
    });
}

/**
 * API: Send message
 * POST /send
 * Body: { to: "number@c.us", message: "text" }
 */
app.post('/send', async (req, res) => {
    try {
        const { to, message } = req.body;
        
        if (!client) {
            return res.status(503).json({ error: 'Client not ready' });
        }
        
        if (!to || !message) {
            return res.status(400).json({ error: 'Missing to or message' });
        }
        
        await client.sendText(to, message);
        
        res.json({ success: true, to, message });
        
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * API: Send image
 * POST /send-image
 * Body: { to: "number@c.us", image: "base64", caption: "text" }
 */
app.post('/send-image', async (req, res) => {
    try {
        const { to, image, caption } = req.body;
        
        if (!client) {
            return res.status(503).json({ error: 'Client not ready' });
        }
        
        const buffer = Buffer.from(image, 'base64');
        const filename = `image_${Date.now()}.jpg`;
        
        await client.sendImage(to, buffer, filename, caption || '');
        
        res.json({ success: true });
        
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * API: Send voice
 * POST /send-voice
 * Body: { to: "number@c.us", audio: "base64" }
 */
app.post('/send-voice', async (req, res) => {
    try {
        const { to, audio } = req.body;
        
        if (!client) {
            return res.status(503).json({ error: 'Client not ready' });
        }
        
        const buffer = Buffer.from(audio, 'base64');
        const filename = `voice_${Date.now()}.ogg`;
        
        await client.sendVoice(to, buffer, filename);
        
        res.json({ success: true });
        
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * API: Download media
 * POST /download
 * Body: { messageId: "true_number@c.us_xxxx" }
 */
app.post('/download', async (req, res) => {
    try {
        const { messageId } = req.body;
        
        if (!client) {
            return res.status(503).json({ error: 'Client not ready' });
        }
        
        const buffer = await client.decryptFile(messageId);
        const base64 = buffer.toString('base64');
        
        res.json({ success: true, media: base64 });
        
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * API: Get contacts
 * GET /contacts
 */
app.get('/contacts', async (req, res) => {
    try {
        if (!client) {
            return res.status(503).json({ error: 'Client not ready' });
        }
        
        const contacts = await client.getAllContacts();
        
        res.json({ contacts });
        
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * API: Get chats
 * GET /chats
 */
app.get('/chats', async (req, res) => {
    try {
        if (!client) {
            return res.status(503).json({ error: 'Client not ready' });
        }
        
        const chats = await client.getAllChats();
        
        res.json({ chats });
        
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * API: Health check
 * GET /health
 */
app.get('/health', (req, res) => {
    res.json({
        status: client ? 'ready' : 'initializing',
        session: sessionName,
        uptime: process.uptime()
    });
});

/**
 * API: Logout
 * POST /logout
 */
app.post('/logout', async (req, res) => {
    try {
        if (!client) {
            return res.status(503).json({ error: 'Client not ready' });
        }
        
        await client.logout();
        
        res.json({ success: true });
        
        process.exit(0);
        
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Start HTTP server
app.listen(port, () => {
    sendEvent('info', { message: `Bridge API listening on port ${port}` });
});

// Initialize Venom Bot
initVenom();

// Handle shutdown
process.on('SIGINT', async () => {
    sendEvent('info', { message: 'Shutting down...' });
    
    if (client) {
        await client.close();
    }
    
    process.exit(0);
});

process.on('SIGTERM', async () => {
    if (client) {
        await client.close();
    }
    
    process.exit(0);
});
