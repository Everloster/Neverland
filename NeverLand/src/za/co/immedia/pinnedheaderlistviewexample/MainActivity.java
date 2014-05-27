package za.co.immedia.pinnedheaderlistviewexample;

import org.jabe.neverland.R;

import za.co.immedia.pinnedheaderlistview.PinnedHeaderListView;

import android.app.Activity;
import android.os.Bundle;
import android.view.Menu;

public class MainActivity extends Activity {

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_pinnedheader);
        PinnedHeaderListView listView = (PinnedHeaderListView) findViewById(R.id.pinnedListView);
        TestSectionedAdapter sectionedAdapter = new TestSectionedAdapter();
        listView.setAdapter(sectionedAdapter);
    }
 
}
