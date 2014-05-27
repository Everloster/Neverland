package org.jabe.neverland.view;

import org.jabe.neverland.activity.MoveViewDemo;

import android.app.Dialog;
import android.content.Context;
import android.graphics.Rect;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.Window;
import android.view.WindowManager;

public class SearchDialog extends Dialog {
	private int lastX, lastY, screenWidth, screenHeight, statusBarHeight;
	private Window window;
	private int lLeft, lRight, lTop, lBottom;
	private static final int OFFSET_DIST_Y = 80;
	private WindowManager.LayoutParams wl;

	public SearchDialog(Context context) {
		super(context);
	}

	public SearchDialog(Context context, int theme, int viewResId) {
		super(context, theme);
		setContentView(viewResId);
		Rect frame = new Rect();
		((MoveViewDemo) context).getWindow().getDecorView()
				.getWindowVisibleDisplayFrame(frame);
		statusBarHeight = frame.top;
		int contentViewTop = ((MoveViewDemo) context).getWindow()
				.findViewById(Window.ID_ANDROID_CONTENT).getTop();

		DisplayMetrics dm = context.getResources().getDisplayMetrics();
		screenWidth = dm.widthPixels;
		screenHeight = dm.heightPixels;
		Log.i("test", "statusBarHeight:" + statusBarHeight + ",contentViewTOp:"
				+ contentViewTop + "..........screenWidth:" + screenWidth
				+ ",screenHeight:" + screenHeight);
		window = getWindow();
		wl = window.getAttributes();
		wl.gravity = Gravity.CENTER;
		wl.width = (int) (screenWidth * 0.88);
		wl.height = (int) (screenHeight * 0.47);
		wl.y += OFFSET_DIST_Y;
		window.setAttributes(wl);

		lLeft = lRight = (screenWidth - wl.width) / 2;
		lTop = (screenHeight - wl.height - statusBarHeight) / 2;
		lBottom = (screenHeight - wl.height - statusBarHeight) / 2;
		Log.i("test", (int) (screenWidth * 0.88) + "........."
				+ (int) (screenHeight * 0.47) + "......." + lLeft + "........"
				+ lTop + "........");
	}

	@Override
	public boolean onTouchEvent(MotionEvent event) {
		int ea = event.getAction();
		switch (ea) {
		case MotionEvent.ACTION_DOWN:
			lastX = (int) event.getRawX();// 获取触摸事件触摸位置的原始X坐标
			lastY = (int) event.getRawY();
			break;
		case MotionEvent.ACTION_MOVE:
			int dx = (int) event.getRawX() - lastX;
			int dy = (int) event.getRawY() - lastY;
			int cx = (int) event.getX();
			int cy = (int) event.getY();
			if (cy > 0 && cx > 0 && cx < wl.width && cy < wl.height) {
				wl.y += dy; // y小于0上移，大于0下移
				wl.x += dx;
				if (wl.x <= -lLeft) {
					wl.x = -lLeft;
				}
				if (wl.x >= lRight) {
					wl.x = lRight;
				}
				if (wl.y <= -lTop) {
					wl.y = -lTop;
				}
				if (wl.y >= lBottom) {
					wl.y = lBottom;
				}
				window.setAttributes(wl);
			}
			lastX = (int) event.getRawX();
			lastY = (int) event.getRawY();
			break;
		case MotionEvent.ACTION_UP:
			break;
		}
		return true;
	}

	@Override
	public void setContentView(int layoutResID) {
		super.setContentView(layoutResID);
	}
}
