package org.jabe.neverland.smallboy;

import android.app.Activity;
import android.content.Context;
import android.view.WindowManager;

public class SmallBoyManager extends IBoyManager {

	private void showBoyWithLayout(Context activity, IBoy boy,
			WindowManager.LayoutParams layoutParams) {
		if (hasBoy(boy)) {
			return;
		}
		addBoy(boy);
		boy.setWindowLayoutParams(layoutParams);
		ViewUtil.addViewToTop(activity, boy.getRootView(), layoutParams);
		boy.init();
	}
	
	private SmallBoyManager() {
		
	}
	
	private static volatile SmallBoyManager instance = null;
	
	public static SmallBoyManager getInstance() {
		synchronized (SmallBoyManager.class) {
			if (instance == null) {
				synchronized (SmallBoyManager.class) {
					instance = new SmallBoyManager();
				}
			}
		}
		return instance;
	}

	private IBoy mDefaultBoy = null;

	public void showDefaultBoy(Activity activity, IBoyAction callback,
			BoyPosition boyPosition) {
		if (mDefaultBoy == null) {
			mDefaultBoy = new FloatTextView(activity);
			mDefaultBoy.setToolBarHight(ViewUtil
					.getWindowVisibleDisplayFrame(activity).top);
			final WindowManager.LayoutParams params = ViewUtil
					.getLeftTopWindowLayout();
			mDefaultBoy.setWindowLayoutParams(params);
			mDefaultBoy.setCallback(callback);
		}
		showBoy(activity, mDefaultBoy, boyPosition);
	}

	public IBoy getDefaultBoy() {
		return mDefaultBoy;
	}

	public void dimissDefaultBoy(Activity activity) {
		if (mDefaultBoy == null) {
			return;
		}
		dismisBoy(activity, mDefaultBoy);
	}

	@Override
	public void showBoy(Context context, IBoy boy, BoyPosition position) {
		if (boy != null && !hasBoy(boy)) {
			final WindowManager.LayoutParams params = boy.getWindowLayoutParams();
			params.x = position.getX();
			params.y = position.getY();
			showBoyWithLayout(context, boy, params);
		}
	}

	@Override
	public void dismisBoy(Context context, IBoy boy) {
		if (boy != null && hasBoy(boy)) {
			ViewUtil.removeViewInTop(context, boy.getRootView());
			removeBoy(boy);
			boy.dismiss();
		}
	}
}
