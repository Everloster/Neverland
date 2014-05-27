package org.jabe.neverland.smallboy;

import java.util.ArrayList;
import java.util.List;

import org.jabe.neverland.R;

import android.content.Context;
import android.graphics.Rect;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.RelativeLayout;

public class FloatAdView extends RelativeLayout implements IBoy {

	public FloatAdView(Context context, AttributeSet attrs) {
		super(context, attrs);
//		mInterceptIds.add(R.id.nmgc_ad_x);
	}

	private WindowManager.LayoutParams mParams;
	private IBoyAction mBoyAction;
	private List<Integer> mInterceptIds = new ArrayList<Integer>();
	
	public interface Callback {
		public void onClosed();
	}

	@Override
	public android.view.WindowManager.LayoutParams getWindowLayoutParams() {
		return mParams;
	}

	@Override
	public void setWindowLayoutParams(
			android.view.WindowManager.LayoutParams layoutParams) {
		mParams = layoutParams;
	}

	@Override
	public void setToolBarHight(int hight) {
		
	}

	@Override
	public void setCallback(IBoyAction action) {
		this.mBoyAction = action;
	}

	@Override
	public View getBoyView() {
		return this;
	}

	@Override
	public void displayMessage(String message) {
		
	}

	@Override
	public int getToolBarHight() {
		return 0;
	}
	
	@Override
	public boolean onInterceptTouchEvent(MotionEvent ev) {
		if (mInterceptIds == null || mInterceptIds.size() == 0) {
			return super.onInterceptTouchEvent(ev);
		}
		final float xf = ev.getX();
        final float yf = ev.getY();
        final float scrolledXFloat = xf + getScrollX();
        final float scrolledYFloat = yf + getScrollY();
        final int scrolledXInt = (int) scrolledXFloat;
        final int scrolledYInt = (int) scrolledYFloat;
        if (hitRect(this, scrolledXInt, scrolledYInt, mInterceptIds)) {
        	mBoyAction.onClose(this);
        } else {
        	mBoyAction.onClick(this);
        }
		return super.onInterceptTouchEvent(ev);
	}
	
	/**
	 * 
	 * 判断某个点是否在 views(ids)里面
	 * 
	 * @param viewGroup
	 * @param scrolledXInt
	 * @param scrolledYInt
	 * @return
	 */
	private boolean hitRect(ViewGroup viewGroup, final int scrolledXInt, final int scrolledYInt, final List<Integer> ids) {
		final int count = viewGroup.getChildCount();
        final Rect frame = new Rect();
        for (int i = 0; i < count; i++) {
        	final View child = viewGroup.getChildAt(i);
        	if ((child.getVisibility() == View.VISIBLE|| child.getAnimation() != null)) {
                child.getHitRect(frame);
                if (frame.contains(scrolledXInt, scrolledYInt)) {
                	if (child instanceof ViewGroup) {
                		boolean hasHit = hitRect((ViewGroup) child, scrolledXInt, scrolledYInt, ids);
                		if (hasHit) {
                			return true;
                		}
                	}
                	if (ids.contains(child.getId())) {
                		return true;
                	}
                }
            }
        }
        return false;
	}

	@Override
	public void init() {
		
	}

	@Override
	public void destroy() {
		
	}

	@Override
	public void showAnimation() {
		
	}

	@Override
	public void dismiss() {
		
	}
}
