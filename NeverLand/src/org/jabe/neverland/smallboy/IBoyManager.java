package org.jabe.neverland.smallboy;

import java.util.HashMap;

import android.content.Context;

public abstract class IBoyManager {

	private HashMap<String, IBoy> mBoyMap = new HashMap<String, IBoy>();

	protected final void addBoy(IBoy boy) {
		mBoyMap.put(boy.toString(), boy);
	}

	protected final void removeBoy(IBoy boy) {
		mBoyMap.remove(boy.toString());
	}

	protected final IBoy getBoy(String name) {
		return mBoyMap.get(name);
	}

	protected final boolean hasBoy(IBoy boy) {
		return mBoyMap.containsKey(boy.toString());
	}

	public abstract void showBoy(Context context, IBoy boy, BoyPosition position);

	public abstract void dismisBoy(Context context, IBoy boy);
}
