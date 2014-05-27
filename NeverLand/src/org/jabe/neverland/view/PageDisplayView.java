package org.jabe.neverland.view;

import org.jabe.neverland.R;

import android.content.Context;
import android.util.AttributeSet;
import android.widget.ImageView;
import android.widget.LinearLayout;

public class PageDisplayView extends LinearLayout{
	
	public PageDisplayView(Context context, AttributeSet attrs) {
		super(context, attrs);
	}

	private void addIndicator(){
		ImageView slidePot = new ImageView(getContext());
		addView(slidePot);
	}
	

	public void setNumOfPages(int num) {
		for(int i = 0; i < num; i++){
			addIndicator();
		}
		changePage(0);
	}

	public void changePage(int page) {
		int childCount = getChildCount();
		for (int i = 0; i < childCount; i++) {
			ImageView point = (ImageView) getChildAt(i);
			if (i == page) {
				point.setBackgroundResource(R.drawable.point_selected);
			} else {
				point.setBackgroundResource(R.drawable.point_default);
			}
		}
	}
}
