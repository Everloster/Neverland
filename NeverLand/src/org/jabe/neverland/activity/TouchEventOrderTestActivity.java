package org.jabe.neverland.activity;

import java.util.concurrent.atomic.AtomicInteger;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.util.AttributeSet;
import android.util.Log;
import android.view.MotionEvent;
import android.widget.LinearLayout;

public class TouchEventOrderTestActivity extends Activity {
	
	public AtomicInteger mAtomicInteger = new AtomicInteger();
	
	private static final String TAG = TouchEventOrderTestActivity.class.getSimpleName();
	
	private static void log(String content)  {
		Log.v(TAG, content);
	}
	
	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
	}

	public class TestLinearLayout extends LinearLayout {

		public TestLinearLayout(Context context, AttributeSet attrs) {
			super(context, attrs);
			// TODO Auto-generated constructor stub
		}

		public TestLinearLayout(Context context) {
			super(context);
			// TODO Auto-generated constructor stub
		}

		@Override
		public boolean dispatchTouchEvent(MotionEvent ev) {
			final Throwable th = new Throwable();
			final StackTraceElement[] stackTraceElements =  th.getStackTrace();
			for (StackTraceElement stackTraceElement : stackTraceElements) {
				stackTraceElement.getMethodName();
			}
			return super.dispatchTouchEvent(ev);
		}
	}
}
