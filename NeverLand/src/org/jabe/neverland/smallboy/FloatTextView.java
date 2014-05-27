package org.jabe.neverland.smallboy;

import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.jabe.neverland.R;
import org.jabe.neverland.util.Util;

import android.content.Context;
import android.graphics.drawable.AnimationDrawable;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.view.WindowManager.LayoutParams;
import android.view.animation.Animation;
import android.view.animation.Animation.AnimationListener;
import android.view.animation.AnimationUtils;
import android.widget.TextView;

public class FloatTextView extends TextView implements IBoy {

	private final String TAG = FloatTextView.class.getSimpleName();
	private WindowManager.LayoutParams params = null;
	private int TOOL_BAR_HIGH = 0;
	private ExecutorService mExecutorService;
	/**
	 * 相对parent的坐标x
	 */
	private float offsetX;
	/**
	 * 相对parent的坐标y
	 */
	private float offsetY;
	/**
	 * 记录初始点击的位置(相对屏幕左上角)x
	 */
	private float firstX;
	/**
	 * 记录初始点击的位置(相对屏幕左上角,减去状态栏的高度)y
	 */
	private float firstY;
	private volatile boolean isMoving = false;
	private WindowManager wm = (WindowManager) getContext()
			.getApplicationContext().getSystemService(ViewUtil.WINDOW_SERVICE);
	/**
	 * 实际点击相对屏幕左上角的位置
	 */
	private float x;
	/**
	 * 实际点击相对屏幕左上角的位置(减去状态栏的高度)
	 */
	private float y;
	private volatile boolean isLeft = true;
	private IBoyAction mCallback;
	private Context mContext;
	private ViewGroup mDialogRoot;
	private TextView mDialog;
	private static final int M_CHANGE_BG = 0x000001;
	private static final int M_GONE = 0x000002;
	
	private Handler mHandler = new Handler() {
		public void dispatchMessage(android.os.Message msg) {
			Bundle data = msg.getData();
			switch (msg.what) {
			case M_CHANGE_BG:
				if (data != null) {
					int resId = data.getInt("bg_resid");
					if (resId != 0) {
						FloatTextView.this.setBackgroundResource(resId);
					}
				}
				break;
			case M_GONE:
				break;
			default:
				break;
			}
		};
	};
	
	private void startBlinkEyeAnimation() {
		final Thread th = new Thread(new Runnable() {
			@Override
			public void run() {
				try {
					new BlinkEyeAnimation().call();
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		});
		mExecutorService.execute(th);
//		this.setBackgroundResource(R.anim.nmgc_loading_boy);
//		final AnimationDrawable animationDrawable = (AnimationDrawable) this.getBackground();
//		animationDrawable.start();
	}
	
	private void startBlinAndWagsAnimation() {
		final Thread th = new Thread(new Runnable() {
			@Override
			public void run() {
				try {
					new BlinkAndWagsAnimation().call();
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		});
		mExecutorService.execute(th);
	}
	
	private void startWagsTailAnimation() {
		final Thread th = new Thread(new Runnable() {
			@Override
			public void run() {
				try {
					new WagsTailAnimation().call();
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		});
		mExecutorService.execute(th);
	}

	public FloatTextView(Context context) {
		super(context);
		this.mContext = context;
		this.setFocusable(true);
		this.setDrawingCacheEnabled(true);
		if (isLeft) {
			changeBG2Left();
		} else {
			changeBG2Right();
		}
		initView();
		initData();
	}
	
	private class BlinkEyeAnimation implements Callable<Object> {

		public int TIME_UNIT = 100;
		private int REPEAT_TIMES = 2; 
		
		@Override
		public Object call() throws Exception {
			initParams();
			Bundle b = new Bundle();
			if (!isMoving) {
				if (isLeft) {
					for (int i = 0; i < REPEAT_TIMES; i++) {
						blinkLeft(b);
					}
				} else {
					for (int i = 0; i < REPEAT_TIMES; i++) {
						blinkRight(b);
					}
				}
			} else {
				for (int i = 0; i < REPEAT_TIMES; i++) {
					blinkMiddle(b);
				}
			}
			return new Object();
		}

		protected void initParams() {
			
		}

		protected void blinkMiddle(Bundle b) throws InterruptedException {
			Util.dout(TAG, "blinkMiddle");
			send(b, R.drawable.nmgc_boy_alert_3, true, TIME_UNIT*2);
			send(b, R.drawable.nmgc_boy_alert_1, true, TIME_UNIT*2);
			send(b, R.drawable.nmgc_boy_alert_2, true, TIME_UNIT*2);
		}
		
		protected void blinkRight(Bundle b) throws InterruptedException {
			Util.dout(TAG, "blinkRight");
			send(b, R.drawable.nmgc_boy_right2, false, TIME_UNIT);
			send(b, R.drawable.nmgc_boy_right3, false, TIME_UNIT);
			send(b, R.drawable.nmgc_boy_right1, false, TIME_UNIT);
		}

		protected void blinkLeft(Bundle b) throws InterruptedException {
			Util.dout(TAG, "blinkLeft");
			send(b, R.drawable.nmgc_boy_left2, false, TIME_UNIT);
			send(b, R.drawable.nmgc_boy_left3, false, TIME_UNIT);
			send(b, R.drawable.nmgc_boy_left1, false, TIME_UNIT);
		}

		protected void send(Bundle b, int id, boolean byMove, int timeUnit) throws InterruptedException {
			Message m;
			m = mHandler.obtainMessage();
			m.what = M_CHANGE_BG;
			m.setData(b);
			Thread.sleep(timeUnit);
			b.putInt("bg_resid", id);
			if (byMove) {// true : 动的时候换背景   false : 不动的时候换背景
				if (isMoving) {
					m.sendToTarget();
				}
			} else {
				if (!isMoving) {
					m.sendToTarget();
				}
			}
		}
	}
	
	private class BlinkAndWagsAnimation extends BlinkEyeAnimation {
		
		@Override
		protected void initParams() {
			TIME_UNIT = 100;
		}
		
		@Override
		protected void blinkLeft(Bundle b) throws InterruptedException {
			send(b, R.drawable.nmgc_boy_left2, false, TIME_UNIT);
			send(b, R.drawable.nmgc_boy_left3, false, TIME_UNIT);
			send(b, R.drawable.nmgc_boy_left4, false, TIME_UNIT);
			send(b, R.drawable.nmgc_boy_left5, false, TIME_UNIT);
			send(b, R.drawable.nmgc_boy_left1, false, TIME_UNIT);
		}
		
		@Override
		protected void blinkRight(Bundle b) throws InterruptedException {
			send(b, R.drawable.nmgc_boy_right2, false, TIME_UNIT);
			send(b, R.drawable.nmgc_boy_right3, false, TIME_UNIT);
			send(b, R.drawable.nmgc_boy_right4, false, TIME_UNIT);
			send(b, R.drawable.nmgc_boy_right5, false, TIME_UNIT);
			send(b, R.drawable.nmgc_boy_right1, false, TIME_UNIT);
		}
	}
	
	private class WagsTailAnimation implements Callable<Object> {

		private int TIME_UNIT = 100;
		private int REPEAT_TIMES = 2; 
		
		@Override
		public Object call() throws Exception {
			Bundle b = new Bundle();
			if (!isMoving) {
				if (isLeft) {
					for (int i = 0; i < REPEAT_TIMES; i++) {
						blinkLeft(b);
					}
				} else {
					for (int i = 0; i < REPEAT_TIMES; i++) {
						blinkRight(b);
					}
				}
			} else {
				for (int i = 0; i < REPEAT_TIMES; i++) {
					blinkMiddle(b);
				}
			}
			return new Object();
		}

		private void blinkMiddle(Bundle b) throws InterruptedException {
			
		}
		
		private void blinkRight(Bundle b) throws InterruptedException {
			Util.dout(TAG, "blinkRight");
			send(b, R.drawable.nmgc_boy_right4, false, TIME_UNIT);
			send(b, R.drawable.nmgc_boy_right5, false, TIME_UNIT);
			send(b, R.drawable.nmgc_boy_right1, false, TIME_UNIT);
		}

		private void blinkLeft(Bundle b) throws InterruptedException {
			Util.dout(TAG, "blinkLeft");
			send(b, R.drawable.nmgc_boy_left4, false, TIME_UNIT);
			send(b, R.drawable.nmgc_boy_left5, false, TIME_UNIT);
			send(b, R.drawable.nmgc_boy_left1, false, TIME_UNIT);
		}

		private void send(Bundle b, int id, boolean byMove, int timeUnit) throws InterruptedException {
			Message m;
			m = mHandler.obtainMessage();
			m.what = M_CHANGE_BG;
			m.setData(b);
			Thread.sleep(timeUnit);
			b.putInt("bg_resid", id);
			if (byMove) {// true : 动的时候换背景   false : 不动的时候换背景
				if (isMoving) {
					m.sendToTarget();
				}
			} else {
				if (!isMoving) {
					m.sendToTarget();
				}
			}
		}
	}
	
	private void initData() {
		mDialogDismiss = AnimationUtils.loadAnimation(mContext,
				android.R.anim.fade_out);
		mDialogDismiss.setAnimationListener(new AnimationListener() {

			@Override
			public void onAnimationStart(Animation animation) {

			}

			@Override
			public void onAnimationRepeat(Animation animation) {

			}

			@Override
			public void onAnimationEnd(Animation animation) {
				mDialogRoot.setVisibility(View.GONE);
			}
		});
		mDialogDismiss.setFillAfter(true);
		mDialogDismiss.setDuration(3500);
		
		mExecutorService = Executors.newCachedThreadPool();
	}

	private void initView() {
		LayoutInflater layoutInflater = LayoutInflater.from(mContext);
		mDialogRoot = (ViewGroup) layoutInflater.inflate(
				R.layout.nmgc_boy_dialog_layout, null);
		mDialog = (TextView) mDialogRoot
				.findViewById(R.id.nmgc_boy_dialog_text);
		mDialog.setWidth(mContext.getResources().getDimensionPixelSize(
				R.dimen.nmgc_boy_dialog_width));
		mDialog.setHeight(mContext.getResources().getDimensionPixelSize(
				R.dimen.nmgc_boy_dialog_height));

	}

	private void changeBG2Left() {
		isLeft = true;
		this.setBackgroundResource(R.drawable.nmgc_boy_left1);
	}

	private void changeBGPressed() {
		isPressed = true;
		if (isLeft) {
			this.setBackgroundResource(R.drawable.nmgc_boy_left1);
		} else {
			this.setBackgroundResource(R.drawable.nmgc_boy_right1);
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
		this.setBackgroundResource(R.drawable.nmgc_boy_right1);
	}

	@Override
	public boolean onTouchEvent(MotionEvent event) {
		// 触摸点相对于屏幕左上角坐标
		x = event.getRawX();
		y = event.getRawY() - TOOL_BAR_HIGH;
		// Log.d(TAG, "------getX: " + event.getX() + "------getY:" +
		// event.getY());
		// Log.d(TAG, "------getRawX: " + event.getRawX() + "------getRawY:"
		// + event.getRawY());
		switch (event.getAction()) {
		case MotionEvent.ACTION_DOWN:
			Log.d(TAG, "ACTION_DOWN");
			mDialogRoot.setVisibility(View.GONE);
			firstX = event.getRawX();
			firstY = event.getRawY() - TOOL_BAR_HIGH;
			offsetX = event.getX() + this.getWidth() / 2;
			offsetY = event.getY() + this.getHeight() / 2;
			break;
		case MotionEvent.ACTION_MOVE:
			// Log.d(TAG, "ACTION_MOVE");
			move();
			break;
		case MotionEvent.ACTION_UP:
			// Log.d(TAG, "ACTION_UP");
			endUp();
			break;
		}
		return true;
	}

	private boolean isPressed = false;

	private void move() {
		// 当水平或者垂直位移超过一定值的时候,才算移动.
		if (!isMoving
				&& (Math.abs(x - firstX) > this.getWidth() / 2 || Math.abs(y
						- firstY) > this.getHeight() / 2)) {
			isMoving = true;
			// 换移动时候的背景.
			changeBG2Mid();
			// 动画
			startBlinkEyeAnimation();
		}
		if (isMoving) {
			// 如果正在移动, 刷新XY坐标
			updatePosition();
		} else {
			// 如果不在移动
			if (!isPressed) {
				// 移动的距离不够,替换成点触背景
				changeBGPressed();
			}
			// 只刷新垂直位置, 目前这里的处理效果不明显, 后面可以做动画.
			updatePositionY();
		}
	}

	private void changeBG2Mid() {
		isPressed = false;
		this.setBackgroundResource(R.drawable.nmgc_boy_alert_2);
	}

	private void endUp() {
		if (isMoving) {
			isMoving = false;
			leftOrRight();
			startEndAnimation();
			mCallback.onMoveEnd();
		} else {
			mCallback.onClick(this);
			changeBGNotPressed();
		}
		offsetX = offsetY = 0;
		isMoving = false;
	}

	private void startEndAnimation() {
		if (new Random().nextBoolean())	 {
			startBlinkEyeAnimation();
		} else {
			startWagsTailAnimation();
		}
	}

	private void leftOrRight() {
		DisplayMetrics metric = new DisplayMetrics();
		wm.getDefaultDisplay().getMetrics(metric);
		int width = metric.widthPixels; // 屏幕宽度（像素）
		if (x > (width / 2)) {
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
	 * 根据偏移更新位置
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

	@Override
	public LayoutParams getWindowLayoutParams() {
		return params;
	}

	@Override
	public void setWindowLayoutParams(LayoutParams layoutParams) {
		this.params = layoutParams;
		x = layoutParams.x;
		y = layoutParams.y;
	}

	@Override
	public void setToolBarHight(int hight) {
		TOOL_BAR_HIGH = hight;
	}

	@Override
	public void setCallback(IBoyAction action) {
		mCallback = action;
	}

	@Override
	public View getBoyView() {
		return this;
	}

	@Override
	public View getRootView() {
		return this;
	}

	private Animation mDialogDismiss;

	@Override
	public void displayMessage(String message) {
		DisplayMetrics metric = new DisplayMetrics();
		wm.getDefaultDisplay().getMetrics(metric);
		int height = metric.heightPixels; // 屏幕高度（像素）
		WindowManager.LayoutParams p = ViewUtil.getLeftTopWindowLayout();
		if ((params.y + getHeight()) > ((height - TOOL_BAR_HIGH) / 2)) {
			mDialogRoot.setPadding(20, 15, 20, 20);
			p.y = params.y - getHeight() - getHeight() / 2;
			if (isLeft) {
				// 左下
				p.x = params.x + getWidth() / 2;
				mDialogRoot
						.setBackgroundResource(R.drawable.nmgc_boy_dialog_left_down);

			} else {
				// 右下
				p.x = params.x - getDimensById(R.dimen.nmgc_boy_dialog_width)
						- getMeasuredWidth();
				mDialogRoot
						.setBackgroundResource(R.drawable.nmgc_boy_dialog_right_down);
			}
		} else {
			mDialogRoot.setPadding(20, 25, 20, 0);
			p.y = params.y + getHeight();
			if (isLeft) {
				// 左上
				p.x = params.x + getMeasuredWidth() / 2;
				mDialogRoot
						.setBackgroundResource(R.drawable.nmgc_boy_dialog_left_up);

			} else {
				// 右上
				p.x = params.x - getDimensById(R.dimen.nmgc_boy_dialog_width)
						- getMeasuredWidth();
				mDialogRoot
						.setBackgroundResource(R.drawable.nmgc_boy_dialog_right_up);
			}
		}
		if (hasAddDialog) {
			ViewUtil.updateTopView(mContext, mDialogRoot, p);
		} else {
			ViewUtil.addViewToTop(mContext, mDialogRoot, p);
			hasAddDialog = true;
		}
		mDialog.setText(message);
		mDialogRoot.setVisibility(View.VISIBLE);
		mDialog.startAnimation(mDialogDismiss);
		startBlinAndWagsAnimation();
	}

	private int getDimensById(int resId) {
		return mContext.getResources().getDimensionPixelSize(resId);
	}

	private boolean hasAddDialog = false;

	@Override
	public int getToolBarHight() {
		return TOOL_BAR_HIGH;
	}

	@Override
	public void init() {
		leftOrRight();
	}

	@Override
	public void destroy() {
		
	}

	@Override
	public void showAnimation() {
		startEndAnimation();
	}

	@Override
	public void dismiss() {
		mDialogRoot.setVisibility(GONE);
		ViewUtil.removeViewInTop(mContext, mDialogRoot);
		hasAddDialog = false;
	}
}
