package org.jabe.neverland.view;

import org.jabe.neverland.util.Util;

import android.content.Context;
import android.util.AttributeSet;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AbsListView;
import android.widget.AbsListView.OnScrollListener;
import android.widget.BaseAdapter;
import android.widget.LinearLayout;
import android.widget.ListView;

public class PullRefreshListView extends ListView implements OnScrollListener {

	public interface OnRefreshListener {
		public void onRefresh();

		public void onStateChange(PullRefreshListView listView, int nowState,
				int oldState);
	}

	public final static int DONE = 3;
	public final static int PULL_To_REFRESH = 1;
	public final static int REFRESHING = 2;
	public final static int RELEASE_To_REFRESH = 0;

	private final static int RATIO = 3;

	private int firstItemIndex;
	private int headContentHeight;
	private int headContentWidth;

	private LinearLayout headView;

	private boolean isRecored;
	private boolean isRefreshable;

	private OnRefreshListener refreshListener;

	private int startY;

	private int state;

	public PullRefreshListView(Context context) {
		super(context);
		init(context);
	}

	public PullRefreshListView(Context context, AttributeSet attrs) {
		super(context, attrs);
		init(context);
	}

	/**
	 * dispatch event(no redraw) 
	 * 
	 * @param newState
	 */
	private void changeHeaderViewByState(int newState) {
		if (refreshListener != null) {
			refreshListener.onStateChange(this, newState, state);
			state = newState;
		} else {
			Util.dout("Error refreshListener is null.");
		}
	}

	private void init(Context context) {

		setCacheColorHint(context.getResources().getColor(
				android.R.color.transparent));
		setOnScrollListener(this);
		state = DONE;
		isRefreshable = false;
	}

	private void measureView(View child) {
		ViewGroup.LayoutParams p = child.getLayoutParams();
		if (p == null) {
			p = new ViewGroup.LayoutParams(ViewGroup.LayoutParams.FILL_PARENT,
					ViewGroup.LayoutParams.WRAP_CONTENT);
		}
		int childWidthSpec = ViewGroup.getChildMeasureSpec(0, 0 + 0, p.width);
		int lpHeight = p.height;
		int childHeightSpec;
		if (lpHeight > 0) {
			childHeightSpec = MeasureSpec.makeMeasureSpec(lpHeight,
					MeasureSpec.EXACTLY);
		} else {
			childHeightSpec = MeasureSpec.makeMeasureSpec(0,
					MeasureSpec.UNSPECIFIED);
		}
		child.measure(childWidthSpec, childHeightSpec);
	}

	private void onRefresh() {
		if (refreshListener != null) {
			refreshListener.onRefresh();
		}
	}

	@Override
	public void onScroll(AbsListView arg0, int firstVisiableItem, int arg2,
			int arg3) {
		firstItemIndex = firstVisiableItem;
	}

	@Override
	public void onScrollStateChanged(AbsListView arg0, int arg1) {

	}

	@Override
	public boolean onTouchEvent(MotionEvent event) {
		int tempY = (int) event.getY();
		if (isRefreshable) {
			switch (event.getAction()) {
			case MotionEvent.ACTION_DOWN:
				if (firstItemIndex == 0 && !isRecored) {
					isRecored = true;
					startY = (int) event.getY();
				}
				break;

			case MotionEvent.ACTION_UP:
				if (firstItemIndex == 0 && state != REFRESHING) {
					if (state == PULL_To_REFRESH) {
						changeHeaderViewByState(DONE);
					}
					if (state == RELEASE_To_REFRESH) {
						changeHeaderViewByState(REFRESHING);
						onRefresh();
					}
				}
				isRecored = false;
				break;

			case MotionEvent.ACTION_MOVE:

				if (!isRecored && firstItemIndex == 0) {
					isRecored = true;
					startY = tempY;
				}

				if (firstItemIndex == 0 && state != REFRESHING && isRecored) {

					if (state == RELEASE_To_REFRESH) {

						setSelection(0);

						if (((tempY - startY) / RATIO < headContentHeight)
								&& (tempY - startY) > 0) {
							changeHeaderViewByState(PULL_To_REFRESH);

						} else if (tempY - startY <= 0) {
							changeHeaderViewByState(DONE);
						} else {
						}
					}
					if (state == PULL_To_REFRESH) {

						setSelection(0);

						if ((tempY - startY) / RATIO >= headContentHeight) {
							changeHeaderViewByState(RELEASE_To_REFRESH);
						} else if (tempY - startY <= 0) {
							changeHeaderViewByState(DONE);
						}
					}

					if (state == DONE) {
						if (tempY - startY > 0) {
							changeHeaderViewByState(PULL_To_REFRESH);
						}
					}
				}
				break;
			}
			//redraw headview
			changHeadViewPadding(tempY);
		}

		return super.onTouchEvent(event);
	}

	private void changHeadViewPadding(int tempY) {
		switch (state) {
		case RELEASE_To_REFRESH:
			headView.setPadding(0,
					(tempY - startY) / RATIO - headContentHeight, 0, 0);
			break;
		case PULL_To_REFRESH:
			headView.setPadding(0, -1 * headContentHeight + (tempY - startY)
					/ RATIO, 0, 0);
			break;
		case REFRESHING:
			headView.setPadding(0, 0, 0, 0);
			break;
		case DONE:
			headView.setPadding(0, -1 * headContentHeight, 0, 0);
			break;
		}
		headView.invalidate();
	}

	/**
	 * when state change awoke this listener
	 * 
	 * @param refreshListener
	 */
	public void setonRefreshListener(OnRefreshListener refreshListener) {
		this.refreshListener = refreshListener;
		isRefreshable = true;
	}

	/**
	 * set header view (force linearlayout) and adapter
	 * 
	 * @param view
	 * @param adapter
	 */
	public void setHeadViewAndAdapter(LinearLayout headView, BaseAdapter adapter) {
		this.headView = headView;
		measureView(headView);
		headContentHeight = headView.getMeasuredHeight();
		headContentWidth = headView.getMeasuredWidth();

		// use padding to hide~~~
		headView.setPadding(0, -1 * headContentHeight, 0, 0);
		headView.invalidate();

		Log.v("size", "width:" + headContentWidth + " height:"
				+ headContentHeight);
		addHeaderView(headView, null, false);
		super.setAdapter(adapter);
	}
	
	/**
	 * interface to change state to DONE
	 */
	public void goneHeadView() {
		//传事件
		changeHeaderViewByState(DONE);
		//redraw
		changHeadViewPadding(0);
	}
}
