import { NextRequest, NextResponse } from 'next/server';
import { 
  exportContext, importContext, listTeleports, exportToFile,
  createTeleportState, formatTeleportContextForPrompt,
  type TeleportState 
} from '@/lib/teleport';

declare global {
  var sableTeleportState: TeleportState | null;
}

function getState(): TeleportState {
  if (!global.sableTeleportState) {
    global.sableTeleportState = createTeleportState();
  }
  return global.sableTeleportState;
}

// GET: List available teleport snapshots
export async function GET() {
  try {
    const snapshots = listTeleports();

    return NextResponse.json({
      success: true,
      snapshots: snapshots.map(s => ({
        id: s.id,
        sourceEnv: s.sourceEnv,
        exportedAt: s.exportedAt,
        dataKeys: s.dataKeys,
        filePath: s.filePath,
      })),
    });
  } catch (error) {
    console.error('[teleport] Error listing teleports:', error);
    return NextResponse.json({ success: false, error: (error as Error).message }, { status: 500 });
  }
}

// POST: Export or import teleport context
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action } = body;
    const state = getState();

    switch (action) {
      case 'export': {
        const { summary, modelOverride } = body;
        const result = exportContext(state, {
          conversationSummary: summary || '',
          modelOverride: modelOverride || undefined,
        });
        global.sableTeleportState = result.state;
        return NextResponse.json({ 
          success: true, 
          context: {
            id: result.context.id,
            sourceEnv: result.context.sourceEnv,
            exportedAt: result.context.exportedAt,
            dataKeys: Object.keys(result.context.data),
            hashCount: Object.keys(result.context.hashes).length,
          },
        });
      }
      case 'import': {
        const { source } = body;
        if (!source || typeof source !== 'string') {
          return NextResponse.json({ success: false, error: 'Source file path is required' }, { status: 400 });
        }
        const result = importContext(state, source);
        global.sableTeleportState = result.state;
        return NextResponse.json({ 
          success: true, 
          imported: {
            id: result.imported.id,
            sourceEnv: result.imported.sourceEnv,
            applied: result.applied,
          },
        });
      }
      default:
        return NextResponse.json({ success: false, error: `Unknown action: ${action}` }, { status: 400 });
    }
  } catch (error) {
    console.error('[teleport] Error:', error);
    return NextResponse.json({ success: false, error: (error as Error).message }, { status: 500 });
  }
}
