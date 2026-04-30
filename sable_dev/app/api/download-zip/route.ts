import { NextResponse } from 'next/server';

/**
 * Download project as a zip file.
 * 
 * Accepts files from the request body (client-side sandbox state)
 * and creates a zip archive using Node.js built-in zlib.
 * 
 * Falls back to reading from the active sandbox if no files provided.
 */

// Simple zip file creator using raw binary (no external deps)
// Uses the "store" method (no compression) for simplicity and speed.
function createZipBuffer(files: Record<string, string>): Buffer {
  const entries: Array<{ name: string; data: Buffer }> = [];
  
  for (const [path, content] of Object.entries(files)) {
    // Skip node_modules, .git etc.
    if (path.includes('node_modules/') || path.includes('.git/') || 
        path.includes('.next/') || path.includes('dist/')) continue;
    entries.push({ name: path, data: Buffer.from(content, 'utf-8') });
  }

  // Build ZIP file (store method, no compression)
  const parts: Buffer[] = [];
  const centralDir: Buffer[] = [];
  let offset = 0;

  for (const entry of entries) {
    const nameBytes = Buffer.from(entry.name, 'utf-8');
    const crc = crc32(entry.data);
    
    // Local file header
    const localHeader = Buffer.alloc(30 + nameBytes.length);
    localHeader.writeUInt32LE(0x04034b50, 0);  // signature
    localHeader.writeUInt16LE(20, 4);           // version needed
    localHeader.writeUInt16LE(0, 6);            // flags
    localHeader.writeUInt16LE(0, 8);            // compression (store)
    localHeader.writeUInt16LE(0, 10);           // mod time
    localHeader.writeUInt16LE(0, 12);           // mod date
    localHeader.writeUInt32LE(crc, 14);         // crc32
    localHeader.writeUInt32LE(entry.data.length, 18);  // compressed size
    localHeader.writeUInt32LE(entry.data.length, 22);  // uncompressed size
    localHeader.writeUInt16LE(nameBytes.length, 26);   // filename length
    localHeader.writeUInt16LE(0, 28);           // extra field length
    nameBytes.copy(localHeader, 30);
    
    parts.push(localHeader, entry.data);
    
    // Central directory entry
    const cdEntry = Buffer.alloc(46 + nameBytes.length);
    cdEntry.writeUInt32LE(0x02014b50, 0);   // signature
    cdEntry.writeUInt16LE(20, 4);           // version made by
    cdEntry.writeUInt16LE(20, 6);           // version needed
    cdEntry.writeUInt16LE(0, 8);            // flags
    cdEntry.writeUInt16LE(0, 10);           // compression
    cdEntry.writeUInt16LE(0, 12);           // mod time
    cdEntry.writeUInt16LE(0, 14);           // mod date
    cdEntry.writeUInt32LE(crc, 16);         // crc32
    cdEntry.writeUInt32LE(entry.data.length, 20);  // compressed size
    cdEntry.writeUInt32LE(entry.data.length, 24);  // uncompressed size
    cdEntry.writeUInt16LE(nameBytes.length, 28);   // filename length
    cdEntry.writeUInt16LE(0, 30);           // extra field length
    cdEntry.writeUInt16LE(0, 32);           // file comment length
    cdEntry.writeUInt16LE(0, 34);           // disk number start
    cdEntry.writeUInt16LE(0, 36);           // internal attributes
    cdEntry.writeUInt32LE(0, 38);           // external attributes
    cdEntry.writeUInt32LE(offset, 42);      // offset of local header
    nameBytes.copy(cdEntry, 46);
    
    centralDir.push(cdEntry);
    offset += localHeader.length + entry.data.length;
  }

  const cdOffset = offset;
  let cdSize = 0;
  for (const cd of centralDir) cdSize += cd.length;

  // End of central directory
  const eocd = Buffer.alloc(22);
  eocd.writeUInt32LE(0x06054b50, 0);           // signature
  eocd.writeUInt16LE(0, 4);                    // disk number
  eocd.writeUInt16LE(0, 6);                    // disk with cd
  eocd.writeUInt16LE(entries.length, 8);       // entries on disk
  eocd.writeUInt16LE(entries.length, 10);      // total entries
  eocd.writeUInt32LE(cdSize, 12);              // cd size
  eocd.writeUInt32LE(cdOffset, 16);            // cd offset
  eocd.writeUInt16LE(0, 20);                   // comment length

  return Buffer.concat([...parts, ...centralDir, eocd]);
}

// CRC-32 lookup table
const crcTable = (() => {
  const table = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let j = 0; j < 8; j++) {
      c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
    }
    table[i] = c;
  }
  return table;
})();

function crc32(data: Buffer): number {
  let crc = 0xFFFFFFFF;
  for (let i = 0; i < data.length; i++) {
    crc = crcTable[(crc ^ data[i]) & 0xFF] ^ (crc >>> 8);
  }
  return (crc ^ 0xFFFFFFFF) >>> 0;
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const files: Record<string, string> = body.files || {};
    
    if (Object.keys(files).length === 0) {
      return NextResponse.json({ error: 'No files provided' }, { status: 400 });
    }
    
    const zipBuffer = createZipBuffer(files);
    const zipBytes = new Uint8Array(zipBuffer);

    return new Response(zipBytes, {
      status: 200,
      headers: {
        'Content-Type': 'application/zip',
        'Content-Disposition': 'attachment; filename="project.zip"',
        'Content-Length': String(zipBuffer.length),
      },
    });
  } catch (error) {
    console.error('[download-zip] Error:', error);
    return NextResponse.json(
      { error: (error as Error).message },
      { status: 500 }
    );
  }
}
