import { NextRequest, NextResponse } from 'next/server';
import { getAvatar, getAvatarCard, renameAvatar, awardXP, awardAchievement, type Avatar } from '@/lib/avatar';

declare global {
  var sableAvatar: Avatar | null;
}

// GET: Retrieve current avatar info
export async function GET() {
  try {
    if (!global.sableAvatar) {
      global.sableAvatar = getAvatar();
    }
    
    const card = getAvatarCard(global.sableAvatar);
    
    return NextResponse.json({
      success: true,
      avatar: {
        id: global.sableAvatar.id,
        displayName: global.sableAvatar.displayName,
        species: global.sableAvatar.bones.species,
        speciesEmoji: global.sableAvatar.bones.speciesEmoji,
        stats: global.sableAvatar.bones.stats,
        rarity: global.sableAvatar.bones.rarity,
        shiny: global.sableAvatar.bones.shiny,
        colorPalette: global.sableAvatar.bones.colorPalette,
        personality: global.sableAvatar.soul.personality,
        level: global.sableAvatar.soul.level,
        xp: global.sableAvatar.soul.xp,
        achievements: global.sableAvatar.soul.achievements,
        card,
      },
    });
  } catch (error) {
    console.error('[avatar] Error getting avatar:', error);
    return NextResponse.json({ success: false, error: (error as Error).message }, { status: 500 });
  }
}

// POST: Actions on avatar (rename, award XP)
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action } = body;

    if (!global.sableAvatar) {
      global.sableAvatar = getAvatar();
    }

    switch (action) {
      case 'rename': {
        const { name } = body;
        if (!name || typeof name !== 'string') {
          return NextResponse.json({ success: false, error: 'Name is required' }, { status: 400 });
        }
        global.sableAvatar = renameAvatar(global.sableAvatar, name);
        return NextResponse.json({ 
          success: true, 
          name: global.sableAvatar.soul.customName 
        });
      }
      case 'xp': {
        const { amount } = body;
        if (typeof amount !== 'number' || amount < 0 || amount > 1000) {
          return NextResponse.json({ success: false, error: 'Invalid XP amount' }, { status: 400 });
        }
        global.sableAvatar = awardXP(global.sableAvatar, amount);
        return NextResponse.json({ 
          success: true, 
          level: global.sableAvatar.soul.level,
          xp: global.sableAvatar.soul.xp,
        });
      }
      case 'achievement': {
        const { achievement } = body;
        if (!achievement || typeof achievement !== 'string') {
          return NextResponse.json({ success: false, error: 'Achievement name is required' }, { status: 400 });
        }
        global.sableAvatar = awardAchievement(global.sableAvatar, achievement);
        return NextResponse.json({ 
          success: true, 
          achievements: global.sableAvatar.soul.achievements,
        });
      }
      default:
        return NextResponse.json({ success: false, error: `Unknown action: ${action}` }, { status: 400 });
    }
  } catch (error) {
    console.error('[avatar] Error:', error);
    return NextResponse.json({ success: false, error: (error as Error).message }, { status: 500 });
  }
}
