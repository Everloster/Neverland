package org.jabe.neverland.util;

import org.jabe.neverland.R;
import org.jabe.neverland.activity.MoveViewDemo;
import org.jabe.neverland.view.FloatTextView;
import org.jabe.neverland.view.SearchDialog;

import android.app.Activity;
import android.content.Context;
import android.graphics.PixelFormat;
import android.graphics.Rect;
import android.view.Gravity;
import android.view.View;
import android.view.WindowManager;
import android.view.WindowManager.LayoutParams;

public class ViewUtil {
	public static final String WINDOW_SERVICE = "window";

	public static WindowManager.LayoutParams addTopView(Context context, View view) {
		WindowManager wm = (WindowManager) context.getApplicationContext()
				.getSystemService(WINDOW_SERVICE);
		WindowManager.LayoutParams params = new WindowManager.LayoutParams();
		params.type = WindowManager.LayoutParams.TYPE_SYSTEM_ALERT
				| WindowManager.LayoutParams.TYPE_SYSTEM_OVERLAY;
		params.flags = LayoutParams.FLAG_NOT_TOUCH_MODAL
				| LayoutParams.FLAG_NOT_FOCUSABLE;
		params.width = WindowManager.LayoutParams.WRAP_CONTENT;
		params.height = WindowManager.LayoutParams.WRAP_CONTENT;
		params.alpha = 100;
		params.gravity = Gravity.LEFT|Gravity.TOP;
		params.format = PixelFormat.RGBA_8888;
		// 以屏幕左上角为原点，设置x、y初始值
		params.x = 0;
		params.y = 0;
		wm.addView(view, params);
		return params;
	}
	
	public static void showSearchUI(final MoveViewDemo context) {
		final SearchDialog dialog = new SearchDialog(context,
				R.style.main_dialog, R.layout.act_moveview_search_dialog);
		dialog.show();
	}

	public static void addTopView(Context context, View view,
			WindowManager.LayoutParams params) {
		WindowManager wm = (WindowManager) context.getApplicationContext()
				.getSystemService(WINDOW_SERVICE);
		View tv = new FloatTextView(context);
		wm.addView(tv, params);
	}

	public static Rect getWindowVisibleDisplayFrame(Activity activity) {
		Rect frame = new Rect();
		activity.getWindow().getDecorView().getWindowVisibleDisplayFrame(frame);
		return frame;
	}
	

	/**
	 * FLAG_NOT_TOUCH_MODAL : view本身可以响应touch事件,不在view范围的内的touch事件也可以被传递.
	 * FLAG_NOT_FOCUSABLE   : view将不获取焦点,点击事件将不被响应.
	 * 
	 * 组合之后: 可以响应touch事件,不能响应key事件.
	 * 
	 * @return
	 */
	public static WindowManager.LayoutParams getLeftTopWindowLayout() {
		WindowManager.LayoutParams params = new WindowManager.LayoutParams();
		params.type = WindowManager.LayoutParams.TYPE_SYSTEM_ALERT
				| WindowManager.LayoutParams.TYPE_SYSTEM_OVERLAY;
		params.flags = LayoutParams.FLAG_NOT_TOUCH_MODAL
				| LayoutParams.FLAG_NOT_FOCUSABLE;
		params.width = WindowManager.LayoutParams.WRAP_CONTENT;
		params.height = WindowManager.LayoutParams.WRAP_CONTENT;
		params.alpha = 100;
		params.gravity = Gravity.LEFT | Gravity.TOP;
		params.format = PixelFormat.RGBA_8888;
		// 以屏幕左上角为原点，设置x、y初始值
		params.x = 0;
		params.y = 0;
		return params;
	}

	public static WindowManager.LayoutParams getCenterWindowParams() {
		WindowManager.LayoutParams params = new WindowManager.LayoutParams();
		params.type = WindowManager.LayoutParams.TYPE_PHONE;
		params.flags = LayoutParams.FLAG_BLUR_BEHIND|0x40000000;
		params.width = WindowManager.LayoutParams.WRAP_CONTENT;
		params.height = WindowManager.LayoutParams.WRAP_CONTENT;
		params.alpha = 100;
		params.gravity = Gravity.CENTER;
		params.format = PixelFormat.RGBA_8888;
		return params;
	}

	public static WindowManager.LayoutParams addViewToTop(Context context,
			View view, WindowManager.LayoutParams params) {
		WindowManager wm = (WindowManager) context.getApplicationContext()
				.getSystemService(WINDOW_SERVICE);
		wm.addView(view, params);
		return params;
	}

	public static void updateTopView(Context context, View view,
			WindowManager.LayoutParams params) {
		WindowManager wm = (WindowManager) context.getApplicationContext()
				.getSystemService(WINDOW_SERVICE);
		wm.updateViewLayout(view, params);
	}

	public static void removeViewInTop(Context context, View view) {
		WindowManager wm = (WindowManager) context.getApplicationContext()
				.getSystemService(WINDOW_SERVICE);
		wm.removeView(view);
	}
}
