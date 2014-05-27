package org.jabe.neverland.view;

import java.util.ArrayList;
import java.util.List;

import org.jabe.neverland.view.data.FingerPathData;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.CornerPathEffect;
import android.graphics.EmbossMaskFilter;
import android.graphics.Paint;
import android.graphics.Path;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.VelocityTracker;
import android.view.View;

public class FingerPaintView extends View {
	public static int FINGER_WIDTH = 14;
	private static final float TOUCH_TOLERANCE = 4;
	private Bitmap mCacheBitmap;
	private Paint mCacheBitmapPaint;//
	private Canvas mCacheCanvas;
	private boolean needCache = false;
	private VelocityTracker mVelocityTracker;
	private static final int VELOCITY_UNITS = 1000;
	private float mCurrentVelocity;
	private static final float STROKEWIDTH_MAX = 30;
	private static final float STROKEWIDTH_MIN = 20;
	private volatile float mCurrentStrokeWidth = STROKEWIDTH_MAX;
	private float mCurrentPressure;

	private Path mCurrentPath;
	private Paint mCurrentPaint;

	private int mBkColor = Color.TRANSPARENT;

	private boolean mIsFingerMoving;
	private boolean mFirstMove = false;
	private ArrayList<Float> mCurFingerPath = new ArrayList<Float>();// 轨迹数据
	private float mX, mY;

	private List<DrawPath> savePath;
	// 记录Path路径的对象

	private static class DrawPath {
		public Path path;// 路径
		public Paint paint;// 画笔
		public float strokeWidth;
		public DrawPath(Path path, Paint paint, float strokeWidth) {
			super();
			this.path = path;
			this.paint = paint;
			this.strokeWidth = strokeWidth;
		}

	}

	public interface FingerPaintListener {
		public void onTouchDown();

		public void onTouchUp();
	}

	protected FingerPaintListener mListern;


	public FingerPaintView(Context c) {
		super(c);
	}

	public FingerPaintView(Context context, AttributeSet attrs) {
		super(context, attrs);
	}

	public void init(int width, int height) {
		mCacheBitmap = Bitmap.createBitmap(getWidth(), getHeight(), Bitmap.Config.ARGB_8888);
		mCacheBitmap.eraseColor(getBackgroundColor());
		mCacheBitmapPaint = new Paint(Paint.DITHER_FLAG);
		mCacheCanvas = new Canvas(mCacheBitmap);

		mCurrentPaint = new Paint();
		mCurrentPaint.setAntiAlias(true);
		mCurrentPaint.setDither(true);
		mCurrentPaint.setColor(0xFF000000);
		mCurrentPaint.setStyle(Paint.Style.STROKE);
		mCurrentPaint.setStrokeJoin(Paint.Join.ROUND);
		mCurrentPaint.setStrokeCap(Paint.Cap.ROUND);
		mCurrentPaint.setStrokeWidth(STROKEWIDTH_MAX);
		mCurrentPaint.setPathEffect(new CornerPathEffect(10));
		savePath = new ArrayList<DrawPath>();
	}

	@Override
	protected void onDraw(Canvas canvas) {
		if (mCacheBitmap == null) {
			init(getWidth(), getHeight());
		} else {
			if (savePath.size() != 0) {
				for (DrawPath drawPath : savePath) {
					mCurrentPaint.setStrokeWidth(drawPath.strokeWidth);
					canvas.drawPath(drawPath.path, mCurrentPaint);
				}
			}
		}
	}

	protected void touch_start(float x, float y) {
		mCurrentPath.moveTo(x, y);
		mX = x;
		mY = y;
		if (mListern != null) {
			mListern.onTouchDown();
		}
	}

	protected void touch_move(float x, float y, boolean isFirstPath) {
		if (isFirstPath) {
			mCurrentStrokeWidth = STROKEWIDTH_MIN;
		}
		float dx = Math.abs(x - mX);
		float dy = Math.abs(y - mY);
		if (dx >= TOUCH_TOLERANCE || dy >= TOUCH_TOLERANCE) {
			float x2 = (x + mX) / 2;
			float y2 = (y + mY) / 2;
			mCurrentPath.quadTo(mX, mY, x2, y2);
			savePath.add(getCurrentDrawPath());
			mCurrentPath = new Path();
			mCurrentPath.moveTo(x2, y2);
			mCurrentPath.quadTo(x2, y2, x, y);
			mX = x;
			mY = y;
		}
		savePath.add(getCurrentDrawPath());
		mCurrentPath = new Path();
		mCurrentPath.moveTo(mX, mY);
		if (isFirstPath) {
			mCurrentStrokeWidth = STROKEWIDTH_MAX;
		}
	}

	private DrawPath getCurrentDrawPath() {
		return new DrawPath(mCurrentPath, mCurrentPaint, mCurrentStrokeWidth);
	}

	protected void touch_up() {
		mCurrentPath.lineTo(mX, mY);
		savePath.add(getCurrentDrawPath());
		mCurrentPath = null;
		mCurrentStrokeWidth = STROKEWIDTH_MAX;
		if (mListern != null) {
			mListern.onTouchUp();
		}
	}

	protected void recordPath(float x, float y) {
		if (mCurFingerPath != null) {
			mCurFingerPath.add(x);
			mCurFingerPath.add(y);
		}
	}

	public FingerPathData getFingerPathData() {
		ArrayList<Float> list = mCurFingerPath;
		int size = list.size();
		if (size > 1) {
			float[] object = new float[size];
			for (int i = 0; i < size; i++) {
				object[i] = list.get(i);
			}
			FingerPathData info = new FingerPathData();
			info.fingerPath = object;
			info.penColor = getPenColor();
			return info;
		}
		return null;
	}

	public int getPenColor() {
		return mCurrentPaint.getColor();
	}

	@Override
	public boolean onTouchEvent(MotionEvent event) {
		float x = event.getX();
		float y = event.getY();
		switch (event.getAction()) {
		case MotionEvent.ACTION_DOWN:
			//初始化速度探测器
			if (mVelocityTracker == null) {
	            mVelocityTracker = VelocityTracker.obtain();
	        }
	        mVelocityTracker.addMovement(event);

			mCurrentPath = new Path();
			mCurrentStrokeWidth = STROKEWIDTH_MAX;
			mCurrentPressure = event.getPressure();

			mIsFingerMoving = true;
			touch_start(x, y);
			invalidate();
			break;
		case MotionEvent.ACTION_MOVE:
			mVelocityTracker.addMovement(event);
			boolean isFirstPath = false;
			if (!mFirstMove) {
				isFirstPath = true;
				mVelocityTracker.computeCurrentVelocity(VELOCITY_UNITS);
				mCurrentVelocity = computeVelocity(mVelocityTracker);
				mFirstMove = true;
				System.out.println("first v:" + mCurrentVelocity);
			} else {
				mVelocityTracker.computeCurrentVelocity(VELOCITY_UNITS);
				final float tempV = computeVelocity(mVelocityTracker);
				if (tempV > mCurrentVelocity) {
					removeStrokeWidth();
				} else if (tempV < mCurrentVelocity) {
					addStrokeWidth();
				}
				final float tempP = event.getPressure();
				if (tempP > mCurrentPressure) {
					addStrokeWidth();
				} else if (tempP < mCurrentPressure) {
					removeStrokeWidth();
				}
				mCurrentPressure = tempP;
				mCurrentVelocity = tempV;
				System.out.println("changed v:" + mCurrentVelocity);
			}
			touch_move(x, y, isFirstPath);
			invalidate();
			break;
		case MotionEvent.ACTION_UP:
			touch_up();
			mIsFingerMoving = false;
			mFirstMove = false;
			invalidate();
			break;
		}
		return true;
	}

	private void addStrokeWidth() {
		if (mCurrentStrokeWidth < STROKEWIDTH_MAX) {
			mCurrentStrokeWidth = mCurrentStrokeWidth + 0.8f;
		}
	}

	private void removeStrokeWidth() {
		if (mCurrentStrokeWidth > STROKEWIDTH_MIN) {
			mCurrentStrokeWidth = mCurrentStrokeWidth - 0.8f;
		}
	}

	private float computeVelocity(VelocityTracker v) {
		return (float) Math.sqrt((v.getXVelocity() * v.getXVelocity() + v.getYVelocity() * v.getYVelocity()));
	}

	/*
	 * 设置线条浮雕效果
	 */
	public void setEmBoss(boolean b) {
		if (mCurrentPaint != null) {
			if (b) {
				mCurrentPaint.setMaskFilter(new EmbossMaskFilter(
						new float[] { 1, 1, 1 }, 0.4f, 6, 3.5f));
			} else {
				mCurrentPaint.setMaskFilter(null);
			}
		}
	}

	/*
	 * 设置背景色
	 */
	public void setBackgroundColor(int color) {
		mBkColor = color;
	}

	public int getBackgroundColor() {
		return mBkColor;
	}

	/*
	 * 设置线条粗细
	 */
	public void setFingerWidth(float width) {
		if (mCurrentPaint != null) {
			mCurrentPaint.setStrokeWidth(width);
		}
	}

	/*
	 * 设置线条颜色
	 */
	public void setPenColor(int color) {
		if (mCurrentPaint != null) {
			mCurrentPaint.setColor(color);
		}
	}

	/*
	 * 获得当前界面绘制的图片bitmap
	 */
	public Bitmap getBitmap() {
		return mCacheBitmap;
	}

	public boolean isFingerMoving() {
		return mIsFingerMoving;
	}

	public ArrayList<Float> getCurFingerPath() {
		return mCurFingerPath;
	}

	public void setListener(FingerPaintListener l) {
		mListern = l;
	}
}
