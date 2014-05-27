package org.jabe.neverland.smallboy;

import java.io.ByteArrayOutputStream;

import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Bitmap.CompressFormat;
import android.graphics.PixelFormat;
import android.graphics.Rect;
import android.util.Base64;
import android.view.Display;
import android.view.Gravity;
import android.view.View;
import android.view.WindowManager;
import android.view.WindowManager.LayoutParams;

public class ViewUtil {
	public static final String WINDOW_SERVICE = "window";

	public static String getBitmapStrBase64(Bitmap bitmap) {
		if (bitmap == null) {
			return "";
		}
		ByteArrayOutputStream baos = new ByteArrayOutputStream();
		bitmap.compress(CompressFormat.JPEG, 100, baos);
		byte[] bytes = baos.toByteArray();
		return Base64.encodeToString(bytes, Base64.DEFAULT);
	}

	/**
	 * FLAG_NOT_TOUCH_MODAL : view本身可以响应touch事件,不在view范围的内的touch事件也可以被传递.
	 * FLAG_NOT_FOCUSABLE : view将不获取焦点,点击事件将不被响应.
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
		params.alpha = 1;
		params.gravity = Gravity.LEFT | Gravity.TOP;
		params.format = PixelFormat.RGBA_8888;
		return params;
	}

	public static WindowManager.LayoutParams getAdLayout(int location) {
		WindowManager.LayoutParams params = new WindowManager.LayoutParams();
		params.type = WindowManager.LayoutParams.TYPE_PHONE;
		params.flags = LayoutParams.FLAG_NOT_TOUCH_MODAL
				| LayoutParams.FLAG_NOT_FOCUSABLE;
		params.width = WindowManager.LayoutParams.WRAP_CONTENT;
		params.height = WindowManager.LayoutParams.WRAP_CONTENT;
		params.alpha = 1;
		switch (location) {
		case 0://上中
			params.gravity = Gravity.TOP | Gravity.CENTER_HORIZONTAL;
			break;
		case 1://上左
			params.gravity = Gravity.TOP | Gravity.LEFT;
			break;
		case 2://上右
			params.gravity = Gravity.TOP | Gravity.RIGHT;
			break;
		case 3://中
			params.gravity = Gravity.CENTER;
			break;
		case 4://左中
			params.gravity = Gravity.LEFT | Gravity.CENTER_VERTICAL;
			break;
		case 5://右中
			params.gravity = Gravity.RIGHT | Gravity.CENTER_VERTICAL;
			break;
		case 6://下中
			params.gravity = Gravity.BOTTOM | Gravity.CENTER_HORIZONTAL;
			break;
		case 7://下左
			params.gravity = Gravity.BOTTOM | Gravity.LEFT;
			break;
		case 8://下右
			params.gravity = Gravity.BOTTOM | Gravity.RIGHT;
			break;
		default:
			params.gravity = Gravity.TOP | Gravity.CENTER_HORIZONTAL;
			break;
		}
		params.format = PixelFormat.RGBA_8888;
		return params;
	}

	public static WindowManager.LayoutParams getCenterWindowParams() {
		WindowManager.LayoutParams params = new WindowManager.LayoutParams();
		params.type = WindowManager.LayoutParams.TYPE_PHONE;
		params.flags = 0x40000000;
		params.width = WindowManager.LayoutParams.WRAP_CONTENT;
		params.height = WindowManager.LayoutParams.WRAP_CONTENT;
		params.alpha = 1;
		params.gravity = Gravity.CENTER;
		params.format = PixelFormat.RGBA_8888;
		return params;
	}

	public static WindowManager.LayoutParams addViewToTop(Context context,
			View view, WindowManager.LayoutParams params) {
		WindowManager wm = (WindowManager) context.getApplicationContext()
				.getSystemService(WINDOW_SERVICE);
		try {
			wm.removeView(view);
		} catch (Exception e) {
		}
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
		try {
			wm.removeView(view);
		} catch (Exception e) {
		}
	}

	public static Rect getWindowVisibleDisplayFrame(Activity activity) {
		Rect frame = new Rect();
		activity.getWindow().getDecorView().getWindowVisibleDisplayFrame(frame);
		return frame;
	}

	public static Display getPhoneDisplay(Context context) {
		WindowManager wm = (WindowManager) context.getApplicationContext()
				.getSystemService(WINDOW_SERVICE);
		return wm.getDefaultDisplay();
	}
}
