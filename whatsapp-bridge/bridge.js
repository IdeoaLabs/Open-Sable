/**
 * WhatsApp Bridge for OpenSable
 *
 * Uses whatsapp-web.js (wwebjs) — the most reliable WhatsApp Web library.
 * Incoming messages → HTTP POST to Python webhook (port 3334).
 * REST API on port 3333 for Python → Bridge outbound (send, media, etc.).
 */

const { Client, LocalAuth, MessageMedia } = require('whatsapp-web.js');
const qrcode = require('qrcode-terminal');
const express = require('express');
const http = require('http');

// ── CLI args ────────────────────────────────────────────────────────
const args = process.argv.slice(2);
const idx = (flag) => args.indexOf(flag);
const sessionName = idx('--session') >= 0 ? args[idx('--session') + 1] : 'opensable';
const port = idx('--port') >= 0 ? parseInt(args[idx('--port') + 1]) : 3333;
const callbackPort = idx('--callback-port') >= 0 ? parseInt(args[idx('--callback-port') + 1]) : 3334;

// ── Express API (Python → Bridge) ──────────────────────────────────
const app = express();
app.use(express.json({ limit: '50mb' }));

// ── POST event to Python webhook ───────────────────────────────────
function postToPython(type, data) {
    const payload = JSON.stringify({ type, data: data || {} });
    const options = {
        hostname: '127.0.0.1',
        port: callbackPort,
        path: '/whatsapp-event',
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload) },
        timeout: 5000,
    };
    const req = http.request(options, (res) => { res.resume(); });
    req.on('error', (err) => {
        process.stderr.write(`[bridge] POST to Python failed: ${err.message}\n`);
    });
    req.write(payload);
    req.end();
}

/** Also write to stdout for early boot events */
function sendEvent(type, data) {
    const line = JSON.stringify({ type, data: data || {} });
    try { process.stdout.write(line + '\n'); } catch (_) {}
    postToPython(type, data || {});
}

// ── WhatsApp Client ─────────────────────────────────────────────────
const client = new Client({
    authStrategy: new LocalAuth({
        clientId: sessionName,
        dataPath: __dirname + '/tokens',
    }),
    puppeteer: {
        headless: false,
        args: [
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-dev-shm-usage',
            '--disable-accelerated-2d-canvas',
            '--no-first-run',
            '--disable-gpu',
        ],
    },
});

// ── Client events ───────────────────────────────────────────────────
client.on('loading_screen', (percent, message) => {
    process.stderr.write(`[wwebjs] Loading: ${percent}% - ${message}\n`);
});

client.on('qr', (qr) => {
    process.stderr.write('[wwebjs] QR code received — scan with your phone\n');
    qrcode.generate(qr, { small: true });
    sendEvent('qr', { qr });
});

client.on('authenticated', () => {
    process.stderr.write('[wwebjs] Authenticated\n');
    sendEvent('authenticated', { session: sessionName });
});

client.on('auth_failure', (msg) => {
    process.stderr.write(`[wwebjs] Auth failure: ${msg}\n`);
    sendEvent('error', { error: `Auth failure: ${msg}` });
});

client.on('ready', () => {
    process.stderr.write('[wwebjs] Client is READY\n');
    sendEvent('ready', { message: 'WhatsApp bot is ready!' });
});

client.on('disconnected', (reason) => {
    process.stderr.write(`[wwebjs] Disconnected: ${reason}\n`);
    sendEvent('disconnected', { reason });
});

// ── Message handling ────────────────────────────────────────────────
client.on('message_create', async (msg) => {
    try {
        const chat = await msg.getChat();
        const contact = await msg.getContact();

        process.stderr.write(
            `[wwebjs] MSG from=${msg.from} fromMe=${msg.fromMe} ` +
            `type=${msg.type} body="${(msg.body || '').substring(0, 50)}"\n`
        );

        postToPython('message', {
            from:       msg.from,
            to:         msg.to,
            body:       msg.body || '',
            type:       msg.type,
            fromMe:     msg.fromMe,
            isGroupMsg: chat.isGroup,
            chatId:     msg.from,
            notifyName: contact.pushname || contact.name || msg.from,
            id:         msg.id._serialized || '',
            timestamp:  msg.timestamp || Math.floor(Date.now() / 1000),
            hasMedia:   msg.hasMedia,
        });
    } catch (error) {
        process.stderr.write(`[wwebjs] message_create error: ${error.message}\n`);
    }
});

// ── REST API endpoints ──────────────────────────────────────────────

app.post('/send', async (req, res) => {
    try {
        const { to, message } = req.body;
        if (!to || !message) return res.status(400).json({ error: 'Missing to or message' });
        await client.sendMessage(to, message);
        res.json({ success: true, to, message });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.post('/send-image', async (req, res) => {
    try {
        const { to, image, caption } = req.body;
        const media = new MessageMedia('image/jpeg', image, `img_${Date.now()}.jpg`);
        await client.sendMessage(to, media, { caption: caption || '' });
        res.json({ success: true });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.post('/send-voice', async (req, res) => {
    try {
        const { to, audio } = req.body;
        const media = new MessageMedia('audio/ogg', audio, `voice_${Date.now()}.ogg`);
        await client.sendMessage(to, media, { sendAudioAsVoice: true });
        res.json({ success: true });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.post('/download', async (req, res) => {
    try {
        const { messageId } = req.body;
        const msg = await client.getMessageById(messageId);
        if (!msg || !msg.hasMedia) return res.status(404).json({ error: 'No media' });
        const media = await msg.downloadMedia();
        res.json({ success: true, media: media.data, mimetype: media.mimetype });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.get('/contacts', async (req, res) => {
    try {
        const contacts = await client.getContacts();
        res.json({ contacts: contacts.map(c => ({ id: c.id._serialized, name: c.pushname || c.name })) });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.get('/chats', async (req, res) => {
    try {
        const chats = await client.getChats();
        res.json({ chats: chats.map(c => ({ id: c.id._serialized, name: c.name, isGroup: c.isGroup })) });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.get('/health', (req, res) => {
    const info = client.info;
    res.json({
        status: info ? 'ready' : 'initializing',
        session: sessionName,
        uptime: process.uptime(),
        phone: info?.wid?._serialized || null,
    });
});

app.post('/logout', async (req, res) => {
    try {
        await client.logout();
        res.json({ success: true });
        setTimeout(() => process.exit(0), 1000);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// ── Start ───────────────────────────────────────────────────────────
app.listen(port, () => {
    process.stderr.write(`[bridge] API on :${port}, callbacks to :${callbackPort}\n`);
});

process.stderr.write('[bridge] Initializing whatsapp-web.js client...\n');
client.initialize();

process.on('SIGINT', async () => {
    try { await client.destroy(); } catch (_) {}
    process.exit(0);
});
process.on('SIGTERM', async () => {
    try { await client.destroy(); } catch (_) {}
    process.exit(0);
});
