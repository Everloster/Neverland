package org.jabe.neverland.activity;

import org.jabe.neverland.R;
import org.jabe.neverland.activity.fragment.ColorFragment;
import org.jabe.neverland.activity.fragment.ColorMenuFragment;
import org.jabe.neverland.activity.fragment.SampleListFragment;

import android.graphics.Canvas;
import android.os.Bundle;
import android.support.v4.app.Fragment;

import com.slidingmenu.lib.SlidingMenu;
import com.slidingmenu.lib.SlidingMenu.CanvasTransformer;

public class SlidingTestActivity extends SlidingBaseActivity {

	private Fragment mContent;
	
	private CanvasTransformer mTransformer = new CanvasTransformer() {
		@Override
		public void transformCanvas(Canvas canvas, float percentOpen) {
			float scale = (float) (percentOpen*0.25 + 0.75);
			canvas.scale(scale, scale, canvas.getWidth()/2, canvas.getHeight()/2);
		}			
	} ;

	@Override
	public void onCreate(Bundle savedInstanceState) {
		// TODO Auto-generated method stub
		super.onCreate(savedInstanceState);
		if (savedInstanceState != null)
			mContent = getSupportFragmentManager().getFragment(savedInstanceState, "mContent");
		if (mContent == null)
			mContent = new ColorFragment(R.color.red);	
		
		
		// set the Above View
		setContentView(R.layout.act_sliding_content_frame);
		getSupportFragmentManager()
		.beginTransaction()
		.replace(R.id.sliding_content_frame, mContent)
		.commit();
		
		// set the Behind View
		setBehindContentView(R.layout.act_sliding_menu_frame);
		getSupportFragmentManager()
		.beginTransaction()
		.replace(R.id.sliding_menu_frame, new ColorMenuFragment())
		.commit();
		
		// customize the SlidingMenu
		getSlidingMenu().setMode(SlidingMenu.LEFT_RIGHT);
		getSlidingMenu().setTouchModeAbove(SlidingMenu.TOUCHMODE_FULLSCREEN);
		getSlidingMenu().setBehindCanvasTransformer(mTransformer);
		
		// set the secondary menu 
		getSlidingMenu().setSecondaryMenu(R.layout.act_sliding_menu_two_frame);
		getSupportFragmentManager()
		.beginTransaction()
		.replace(R.id.sliding_menu_two_frame, new SampleListFragment())
		.commit();
		
	}
	
	@Override
	public void onSaveInstanceState(Bundle outState) {
		super.onSaveInstanceState(outState);
		getSupportFragmentManager().putFragment(outState, "mContent", mContent);
	}
	
	public void switchContent(Fragment fragment) {
		mContent = fragment;
		getSupportFragmentManager()
		.beginTransaction()
		.replace(R.id.sliding_content_frame, fragment)
		.commit();
		getSlidingMenu().showContent();
	}
	
}
