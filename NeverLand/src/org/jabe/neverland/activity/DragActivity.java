package org.jabe.neverland.activity;

import org.jabe.neverland.R;
import org.jabe.neverland.view.DragLayer;
import org.jabe.neverland.view.DragLayer.PageListener;

import android.app.ActivityGroup;
import android.app.LocalActivityManager;
import android.content.Intent;
import android.os.Bundle;
import android.view.Window;

public class DragActivity extends ActivityGroup {

	DragLayer root;

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.act_drag_main);

		initView();

	}

	private void initView() {
		root = (DragLayer) findViewById(R.id.root);
		LocalActivityManager lam = getLocalActivityManager();

		Window w1 = lam.startActivity("1", new Intent(this,
				PullRefreshActivity.class));
		Window w2 = lam.startActivity("2", new Intent(this,
				FastScrollerActivity.class));

		root.addView(w1.getDecorView());
		root.addView(w2.getDecorView());
		root.setPageListener(new PageListener() {
			@Override
			public void page(int page) {
				
			}
		});
	}

}
