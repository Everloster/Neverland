package org.jabe.neverland.activity;

import org.jabe.neverland.R;
import org.jabe.neverland.util.ViewUtil;

import android.app.Activity;
import android.os.Bundle;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.Button;

public class MoveViewDemo extends Activity {
    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.act_moveview_main);
        Button haha = (Button)findViewById(R.id.haha);
        haha.setOnClickListener(new OnClickListener() {
			
			@Override
			public void onClick(View v) {
				// TODO Auto-generated method stub
				ViewUtil.showSearchUI(MoveViewDemo.this);
			}
		});
    }
}