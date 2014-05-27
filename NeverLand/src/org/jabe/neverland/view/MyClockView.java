package org.jabe.neverland.view;

import java.util.Calendar;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Point;
import android.util.Log;
import android.view.KeyEvent;
import android.view.MotionEvent;
import android.view.View;
import android.view.View.OnTouchListener;

public class MyClockView extends View implements OnTouchListener{

	private static final String TAG = "MyClockView";
	

	private Bitmap mClockBitmap;	// 时钟底盘
	private Bitmap mHourBitmap;		// 时针
	private Bitmap mMinuteBitmap;	// 分针

	// 时钟的位置（相对于视图）
	private int clockX = 0, clockY = 0;	
	
	// 时钟中心点位置（相对于视图）
	private int clockCenterX = 0, clockCenterY = 0;
	
	// 指针指向12点钟方向时指针向下的偏移量
	private int mHourOffsetY = 0, mMinuteOffsetY = 0;
	
	// 时针位置（相对于时钟中心点）
	private int mHourPosX = 0, mHourPosY = 0;
	
	// 分针位置（相对于时钟中心点）
	private int mMinutePosX = 0, mMinutePosY = 0;
	
	// 是否初始化完毕
	private boolean bInitComplete = false;
	
	// 时钟当前时间
	private MyTime mCurTime;
	
	public MyClockView(Context context) {
		super(context);
		// TODO Auto-generated constructor stub
		mCurTime = new MyTime();
		
		setOnTouchListener(this);
	}
	
	/**
	 * @param clock
	 * @param hour
	 * @param minute
	 * 
	 * 初始化视图
	 */
	public void init(Bitmap clock, Bitmap hour, Bitmap minute){
		mClockBitmap = clock;
		mHourBitmap = hour;
		mMinuteBitmap = minute;
		
		calcPointPosition();
		
		calcCenter();
		
		bInitComplete = true;
		
		mCurTime.initBySystem();
	}
	
	/**
	 * @param cx	中心点相对于时钟图片的X偏移量
	 * @param cy    中心点相对于时钟图片的Y偏移量
	 * 
	 * 设置时钟中心点位置（指针要绕着这个点旋转）
	 */
	public void setCenter(int cx, int cy){
		clockCenterX = clockX + cx;
		clockCenterY = clockY + cy;
	}
	
	/**
	 * @param hourOffset
	 * @param minuteOffset
	 * 
	 * 设置指针12点钟方向时向下的偏移量
	 */
	public void setPointOffset(int hourOffset, int minuteOffset){
		mHourOffsetY = hourOffset;
		mMinuteOffsetY = minuteOffset;
		
		calcPointPosition();
	}
	
	/**
	 * 计算时钟中心点位置（没有用setCenter设置时采用此默认值）
	 */
	public void calcCenter(){
		if (mClockBitmap != null){
			clockCenterX = clockX + mClockBitmap.getWidth()/2;
			clockCenterY = clockY + mClockBitmap.getHeight()/2;
		}
	}

	/**
	 * 计算指针位置
	 */
	public void calcPointPosition(){
		if (mHourBitmap != null){
			int w = mHourBitmap.getWidth();
			int h = mHourBitmap.getHeight();
			
			mHourPosX =  -w/2;
			mHourPosY = -h + mHourOffsetY;
			
		}
		
		if (mMinuteBitmap != null){
			int w = mMinuteBitmap.getWidth();
			int h = mMinuteBitmap.getHeight();
			
			mMinutePosX = -w/2;
			mMinutePosY = -h + mMinuteOffsetY;
		}
	}
	
	@Override
	protected void onDraw(Canvas canvas) {
		// TODO Auto-generated method stub
		super.onDraw(canvas);

		
		canvas.drawColor(Color.WHITE);
		
		if (!bInitComplete){
			return ;
		}
			
		drawClock(canvas);
		
		drawHour(canvas);
		
		drawMinute(canvas);
		
		Paint paint = new Paint();
		paint.setColor(Color.RED);
		
		canvas.drawLine(0, clockCenterY, canvas.getWidth(), clockCenterY, paint);
		
		canvas.drawLine(clockCenterX, 0, clockCenterX, canvas.getHeight(), paint);
		
	}
	
	
	/**
	 * @param canvas
	 * 绘制时钟
	 */
	public void drawClock(Canvas canvas){
		if (mClockBitmap == null){
			return ;
		}
		
		canvas.drawBitmap(mClockBitmap, clockX, clockY, null);
	}
	
	
	/**
	 * @param canvas
	 * 绘制时针
	 */
	private void drawHour(Canvas canvas){
		if (mHourBitmap == null){
			return ;
		}
		
		canvas.save();
		
		canvas.translate(clockCenterX, clockCenterY);
		
		
		canvas.rotate(mCurTime.mHourDegree);
		
		Paint paint = new Paint();
		paint.setAntiAlias(true);
		
		canvas.drawBitmap(mHourBitmap, mHourPosX, mHourPosY, paint);
		
		canvas.restore();
	}
	
	/**
	 * @param canvas
	 * 绘制分针
	 */
	public void drawMinute(Canvas canvas){
		if (mMinuteBitmap == null){
			return ;
		}
		
		canvas.save();
		
		canvas.translate(clockCenterX, clockCenterY);
		
		
		
		canvas.rotate(mCurTime.mMinuteDegree);
		
		Paint paint = new Paint();
		paint.setAntiAlias(true);
		canvas.drawBitmap(mMinuteBitmap, mMinutePosX, mMinutePosY, paint);
		
		canvas.restore();
	}
	

	/**
	 * @author genius
	 * 管理当前时钟所表示的时间
	 */
	class MyTime{
		
		
		int mHour = 0;				
		int mMinute = 0;

		int mHourDegree = 0;				//  时针偏移量（相对于Y轴正半轴顺时针夹角，参考坐标系原点为时钟中心点，Y轴向上）
		int mMinuteDegree = 0;				//  分针偏移量（相对于Y轴正半轴顺时针夹角，参考坐标系原点为时钟中心点，Y轴向上）
		int mPreDegree = 0;					//  上次分针偏移量
		
		private Calendar mCalendar;
	
		
		/**
		 * 根据系统时间更新相关变量
		 */
		public void initBySystem(){
			long time = System.currentTimeMillis(); 		
	        mCalendar = Calendar.getInstance();
	        mCalendar.setTimeInMillis(time);     
	        
	        mHour = mCalendar.get(Calendar.HOUR_OF_DAY);
	        mMinute = mCalendar.get(Calendar.MINUTE);
	        
	        calcDegreeByTime();
	       
		}
		
		/**
		 * 根据mHour，mMinute计算指针偏移量
		 */
		public void calcDegreeByTime(){
			mMinuteDegree = mMinute * 6;
			mPreDegree = mMinuteDegree;
			mHourDegree = (mHour % 12) * 30 + mMinuteDegree / 12;
		}
		
		/**
		 * @param bFlag  是否校正指针角度（ACTION_UP 时要校正）
		 * 
		 * 根据变化后的mMinuteDegree更新表示时间
		 */
		public void calcTime(boolean bFlag){
			if (mMinuteDegree >= 360){
				mMinuteDegree -= 360;
			}
			
			if (mMinuteDegree < 0){
				mMinuteDegree += 360;
			}
			
			mMinute = (int) ((mMinuteDegree / 360.0) * 60);
			
			
			if (deasil()){
				if (mMinuteDegree < mPreDegree){
					mHour += 1;
					mHour %= 24;
				}
			}else{
				if (mMinuteDegree > mPreDegree){
					mHour -= 1;
					if (mHour < 0){
						mHour += 24;
					}
				}
			}
			
			mHourDegree = (mHour % 12) * 30 + mMinuteDegree / 12;
			
			mPreDegree = mMinuteDegree;
			
			Log.i(TAG, "mHourDegree = " + mHourDegree + ", mMinuteDegree = " + mMinuteDegree);
			
			if (bFlag){				
				calcDegreeByTime();
			}
			
		}
		
		/**
		 * @return
		 * ACTION_MOVE时判断是否为顺时针旋转
		 */
		public boolean deasil(){
			if (mMinuteDegree >= mPreDegree){
				if (mMinuteDegree - mPreDegree < 180){
					return true;
				}
				return false;
			}else{
				if (mPreDegree - mMinuteDegree > 180){
					return true;
				}
				
				return false;
			}
		}

	}


	/**
	 * @param x     	
	 * @param y
	 * @param flag			是否校正指针角度（ACTION_UP 时要校正）
	 * 
	 * 根据事件坐标更新表示时间
	 */
	public void calcDegree(int x, int y, boolean flag){
		int rx = x - clockCenterX;
		int ry = - (y - clockCenterY);
		
		Point point = new Point(rx, ry);
		
		mCurTime.mMinuteDegree = MyDegreeAdapter.GetRadianByPos(point);
		mCurTime.calcTime(flag);
	}

	@Override
	public boolean onTouch(View v, MotionEvent event) {
		// TODO Auto-generated method stub
		
		switch(event.getAction()){
			case MotionEvent.ACTION_DOWN:
			
				calcDegree((int)event.getX(), (int)event.getY(), false);
				postInvalidate();
				
				break;
			case MotionEvent.ACTION_MOVE:
				
				calcDegree((int)event.getX(), (int)event.getY(), false);
				postInvalidate();
				
				break;
			case MotionEvent.ACTION_UP:
				
				calcDegree((int)event.getX(), (int)event.getY(), true);
				postInvalidate();
				
				break;
		}
	
		return true;
	}
	
	
	/* (non-Javadoc)
	 * @see android.view.View#onKeyDown(int, android.view.KeyEvent)
	 * 
	 * 可按左右键调整分针角度
	 */
	@Override
	public boolean onKeyDown(int keyCode, KeyEvent event) {
		// TODO Auto-generated method stub
		
		switch(keyCode){
		case KeyEvent.KEYCODE_DPAD_LEFT:
			mCurTime.mMinuteDegree -= 6;
			mCurTime.calcTime(false);
			
			postInvalidate();
			break;
		case KeyEvent.KEYCODE_DPAD_RIGHT:
			mCurTime.mMinuteDegree += 6;
			mCurTime.calcTime(false);
			
			postInvalidate();
			break;
		default:
			break;
		}

		return super.onKeyDown(keyCode, event);
	}

}
