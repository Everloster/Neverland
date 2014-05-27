package org.jabe.neverland.view;


import org.jabe.neverland.R;
import org.jabe.neverland.util.ViewUtil;

import android.content.Context;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.MotionEvent;
import android.view.WindowManager;
import android.widget.TextView;

public class FloatTextView extends TextView {

	public interface Callback {
		public void onClick(final FloatTextView floatTextView);
	}

	private final String TAG = FloatTextView.class.getSimpleName();
	public static WindowManager.LayoutParams params;
	public static int TOOL_BAR_HIGH = 0;
	private float offsetX;
	private float offsetY;
	private float firstX;
	private float firstY;
	private boolean isMoving = false;
	private WindowManager wm = (WindowManager) getContext()
			.getApplicationContext().getSystemService(ViewUtil.WINDOW_SERVICE);
	private float x;
	private float y;
	private boolean isLeft = true;
	private Callback mCallback = null;

	public void setCallback(Callback callback) {
		this.mCallback = callback;
	}

	public FloatTextView(Context context) {
		super(context);
		changeBG2Left();
	}

	private void changeBG2Left() {
		isLeft = true;
		this.setBackgroundResource(R.drawable.assist_anzai_left_green);
	}

	private void changeBGPressed() {
		isPressed = true;
		if (isLeft) {
			this.setBackgroundResource(R.drawable.assist_anzai_pressed_left_green);
		} else {
			this.setBackgroundResource(R.drawable.assist_anzai_pressed_right_green);
		}
	}

	private void changeBGNotPressed() {
		isPressed = false;
		if (isLeft) {
			changeBG2Left();
		} else {
			changeBG2Right();
		}
	}

	private void changeBG2Right() {
		isLeft = false;
		this.setBackgroundResource(R.drawable.assist_anzai_right_green);
	}

	@Override
	public boolean onTouchEvent(MotionEvent event) {
		// 触摸点相对于屏幕左上角坐标
		x = event.getRawX();
		y = event.getRawY() - TOOL_BAR_HIGH;
		Log.d(TAG, "------getX: " + event.getX() + "------getY:" + event.getY());
		Log.d(TAG, "------getRawX: " + event.getRawX() + "------getRawY:"
				+ event.getRawY());
		switch (event.getAction()) {
		case MotionEvent.ACTION_DOWN:
			Log.d(TAG, "ACTION_DOWN");
			firstX = event.getRawX();
			firstY = event.getRawY() - TOOL_BAR_HIGH;
			offsetX = event.getX();
			offsetY = event.getY() + this.getHeight() / 2;
			break;
		case MotionEvent.ACTION_MOVE:
			Log.d(TAG, "ACTION_MOVE");
			move();
			break;
		case MotionEvent.ACTION_UP:
			Log.d(TAG, "ACTION_UP");
			endUp();
			break;
		}
		return true;
	}

	private boolean isPressed = false;

	private void move() {
		if (!isMoving && (Math.abs(x - firstX) > this.getWidth()/2 || Math.abs(y - firstY) > this.getHeight()/2)) {
			isMoving = true;
			changeBG2Mid();
		}
		if (isMoving) {
			updatePosition();
		} else {
			if (!isPressed) {
				changeBGPressed();
			}
			updatePositionY();
		}
	}

	private void changeBG2Mid() {
		isPressed = false;
		this.setBackgroundResource(R.drawable.assist_anzai_middle_green);
	}

	private void endUp() {
		if (isMoving) {
			leftOrRight();
		} else {
			mCallback.onClick(this);
			changeBGNotPressed();
		}
		offsetX = offsetY = 0;
		isMoving = false;
	}

	private void leftOrRight() {
		DisplayMetrics metric = new DisplayMetrics();
		wm.getDefaultDisplay().getMetrics(metric);
		int width = metric.widthPixels; // 屏幕宽度（像素）
		if ((int) (x - offsetX) > width / 2) {
			params.x = width;
			changeBG2Right();
		} else {
			params.x = 0;
			changeBG2Left();
		}
		params.y = (int) (y - offsetY);
		wm.updateViewLayout(this, params);
	}

	/**
	 * 更新浮动窗口位置参数
	 */
	private void updatePosition() {
		params.x = (int) (x - offsetX);
		params.y = (int) (y - offsetY);
		wm.updateViewLayout(this, params);
	}

	private void updatePositionY() {
		params.y = (int) (y - offsetY);
		wm.updateViewLayout(this, params);
	}
}
